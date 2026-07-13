"""
tdvp_core.py

【このファイルの現在の役割】
このファイル自身の時間発展ドライバ（下の "Route A" 一式）は使われていない
（規約バグあり）。実際に使われているのは pack_state/unpack_state,
build_H_emb_fast, calc_Lmbdac_fast, setup_frequency_grid,
compute_fdaggerc_sp, _best_phi_from_degenerate などのヘルパー関数群で、
これらは正しい実装である td_gGA_solver.py から import されて使われている。
以下のドキュメントは元の（未使用の）ドライバの説明として残してある。

--- 以下、旧ドキュメント ---
Full TD-gGA solver (Route A, Λ=0 gauge): simultaneously evolves |Phi(t)> and n_ab(omega,t).
Implements Eqs.(16)-(21) of Guerci, Capone, Lanata, PRR 5, L032023 (2023).

Route A: Λ=0 gauge — the TDVP-consistent formulation matching the paper directly.
  Initial condition:  n0(ω) = f(ω·RR†)
  ODE for n:          dn/dt = -iω [n, RR†]       (no Λ term)
  Lmbdac:             Lc = Lc_fast               (no -Λ correction)

State vector Y = (Re(Phi), Im(Phi), Re(n(omega)), Im(n(omega)))
Equations of motion:
  [Eq.16]  d|Phi>/dt = -i H_emb(D(t), Lc(t)) |Phi>
  [Eq.17]  dn_ab/dt  = -i*omega * [R_b R*_c n_ac - R*_a R_c n_cb]
Constraints solved at each step:
  [Eq.5]   Delta = integral rho(omega) n(omega) domega
  [Eq.20]  R = [Delta(1-Delta)]^{-1/2} @ fdaggerc_sp  (from |Phi>)
  [Eq.18]  D = [Delta(1-Delta)]^{-1/2} @ (R^T B_mat)^T
  [Eq.19]  Lambda^c via calc_Lmbdac (Lambda=0; TDVP energy conservation holds)
"""
import numpy as np
import scipy.sparse as sp
from math import comb
from numpy.polynomial.legendre import leggauss
import convenience_routines as cr


# ============================================================
# Frequency grid
# ============================================================
def setup_frequency_grid(N_freq):
    """Gauss-Legendre quadrature on [-1,1] with Bethe semicircular DOS."""
    xi, wi = leggauss(N_freq)
    rho = (2.0 / np.pi) * np.sqrt(np.maximum(0.0, 1.0 - xi**2))
    return xi, wi, rho


# ============================================================
# State pack / unpack
# ============================================================
def pack_state(Phi, n_omega):
    return np.concatenate([np.real(Phi), np.imag(Phi),
                           np.real(n_omega).ravel(),
                           np.imag(n_omega).ravel()])


def unpack_state(y, dim_Phi, N_freq, nqspo):
    Phi = y[:dim_Phi] + 1j * y[dim_Phi:2 * dim_Phi]
    r = 2 * dim_Phi
    sz = N_freq * nqspo * nqspo
    n_re = y[r:r + sz].reshape(N_freq, nqspo, nqspo)
    n_im = y[r + sz:r + 2 * sz].reshape(N_freq, nqspo, nqspo)
    return Phi, n_re + 1j * n_im


# ============================================================
# Prepare parameters (precompute operators once)
# ============================================================
def prepare_full_params(ga_obj, U_final, N_freq=100):
    """
    Precompute all operators needed for time evolution.
    Expensive disk I/O happens here once; the ODE loop is disk-free.
    """
    ed = ga_obj.imp_solver
    dim_Phi = ed.hsize_half
    nqspo = ga_obj.nqspo
    n_phys = ed.n_phys_orb   # 2 (spin-up + spin-dn physical)
    n_bath = ed.n_bath_orb   # 6 (spin-bath)
    imp_nr = ed.impurity_nr
    imp_type = ed.impurity_type

    ioff = sum(int(comb(ed.n_tot_orb, i)) for i in range(ed.n_half))
    iend = ioff + dim_Phi

    print("  Loading bath-phys operators...")
    # op_bp[i,j] = f†_{bath,i} c_{phys,j}  shape (n_bath, n_phys, dim, dim)
    op_bp = np.zeros((n_bath, n_phys, dim_Phi, dim_Phi))
    for i in range(n_bath):
        for j in range(n_phys):
            fname = f"bath-phys_imp-{imp_nr}_{imp_type}_op+{i}-{j}.npz"
            op_bp[i, j] = sp.load_npz(fname).toarray()

    print("  Loading bath-bath operators...")
    # op_bb[i,j] = f_j f†_i  (lreverse=True)  shape (n_bath, n_bath, dim, dim)
    op_bb = np.zeros((n_bath, n_bath, dim_Phi, dim_Phi))
    for i in range(n_bath):
        for j in range(n_bath):
            fname = f"bath-bath_imp-{imp_nr}_{imp_type}_op+{i}-{j}.npz"
            op_bb[i, j] = sp.load_npz(fname).toarray()

    print("  Building double-occupancy and orbital operators...")
    FH = ed.build_creation_ops()
    n_up_full = FH[0] @ FH[0].conj().T
    n_dn_full = FH[1] @ FH[1].conj().T
    op_docc = (n_up_full @ n_dn_full)[ioff:iend, ioff:iend].toarray()

    op_n_orb_list = []
    for i in range(nqspo + 1):
        n_up = FH[2 * i] @ FH[2 * i].conj().T
        n_dn = FH[2 * i + 1] @ FH[2 * i + 1].conj().T
        op_n_orb_list.append((n_up + n_dn)[ioff:iend, ioff:iend].toarray())

    # ---- H_emb basis operators ----
    # M_D[p] = (f†_{2p} c_0 + f†_{2p+1} c_1) + H.c.
    print("  Precomputing H_emb basis operators...")
    M_D = np.zeros((nqspo, dim_Phi, dim_Phi))
    for p in range(nqspo):
        tmp = op_bp[2 * p, 0] + op_bp[2 * p + 1, 1]
        M_D[p] = tmp + tmp.T  # + H.c. (operators are real)

    # M_Lc_full[p,q] = f_{2q} f†_{2p} + f_{2q+1} f†_{2p+1}  (full bath-bath matrix)
    # Covers off-diagonal Lmbdac (Lmbdac is generally non-diagonal since H_list includes
    # off-diagonal basis elements from generate_orthonormal_basis).
    M_Lc_full = np.zeros((nqspo, nqspo, dim_Phi, dim_Phi))
    for p in range(nqspo):
        for q in range(nqspo):
            M_Lc_full[p, q] = op_bb[2 * p, 2 * q] + op_bb[2 * p + 1, 2 * q + 1]

    # H_const: fixed part (H_phys from H1 + two-body U + spin penalty), no D, no Lc
    print("  Building constant H_emb part...")
    D_zero = np.zeros((nqspo, 1))
    Lc_zero = np.zeros((nqspo, nqspo))
    H1 = np.array([[-U_final / 2.0]])
    H_const = ed.build_Hemb(D_zero, H1, Lc_zero, U_final).toarray()
    # H_const for initial state SC loop (uses U_initial = ga_obj.U, not U_final)
    H1_init = np.array([[-ga_obj.U / 2.0]])
    H_const_init = ed.build_Hemb(D_zero, H1_init, Lc_zero, ga_obj.U).toarray()

    # ---- Frequency grid ----
    omega_arr, weights, rho_arr = setup_frequency_grid(N_freq)

    # Precompute stacked H_list for fast Lmbdac (avoid 9x redundant eigh)
    H_list = ga_obj.H_list
    H_list_arr = np.stack([h.T for h in H_list])  # (n_basis, nqspo, nqspo)

    # Compute P_npz2imp: permutation from npz bath ordering to imp_solver/fix_gauge ordering.
    # Needed to convert R_t (npz basis) before calling ga_obj.calc_Z(ga_obj.Lmbda, R_t).
    Phi_0_tmp = ed.eig_vec.copy()
    Phi_c_tmp = np.conj(Phi_0_tmp)
    fdc_npz_0 = np.array([[0.5 * (np.real(Phi_c_tmp @ (op_bp[2*i, 0] @ Phi_0_tmp)) +
                                   np.real(Phi_c_tmp @ (op_bp[2*i+1, 1] @ Phi_0_tmp)))]
                          for i in range(nqspo)])
    ffdagger_npz_0 = np.array([[0.5 * (np.real(Phi_c_tmp @ (op_bb[2*i, 2*j] @ Phi_0_tmp)) +
                                        np.real(Phi_c_tmp @ (op_bb[2*i+1, 2*j+1] @ Phi_0_tmp)))
                                 for j in range(nqspo)]
                                for i in range(nqspo)])
    R_npz_0 = (fdc_npz_0.T @ cr.funcMat(ffdagger_npz_0, cr.denR)).T
    R_imp_0 = (ed.fdaggerc.T @ cr.funcMat(ed.ffdagger, cr.denR)).T
    idx_npz = np.argsort(-np.abs(R_npz_0[:, 0]))
    idx_imp = np.argsort(-np.abs(R_imp_0[:, 0]))
    P_npz2imp = np.zeros((nqspo, nqspo))
    for k in range(nqspo):
        P_npz2imp[idx_imp[k], idx_npz[k]] = 1.0

    params = {
        'dim_Phi': dim_Phi,
        'Phi_0': ed.eig_vec.copy(),
        'nqspo': nqspo,
        'n_phys': n_phys,
        'n_bath': n_bath,
        'op_bp': op_bp,
        'op_bb': op_bb,
        'M_D': M_D,
        'M_Lc_full': M_Lc_full,
        'H_const': H_const,
        'H_const_init': H_const_init,
        'op_docc': op_docc,
        'op_n_orb_list': op_n_orb_list,
        'omega_arr': omega_arr,
        'weights': weights,
        'rho_arr': rho_arr,
        'N_freq': N_freq,
        'H_list': H_list,
        'H_list_arr': H_list_arr,
        'P_npz2imp': P_npz2imp,
        'Lmbda_npz': np.zeros((nqspo, nqspo)),  # Route A: Λ=0 (TDVP consistent)
    }
    print(f"  Done. dim_Phi={dim_Phi}, nqspo={nqspo}, N_freq={N_freq}")
    return params


# ============================================================
# Build H_emb from basis operators (no disk I/O)
# ============================================================
def build_H_emb_fast(D_sp, Lmbdac, H_const, M_D, M_Lc_full):
    """
    Assemble H_emb = H_const + sum_p D_p M_D[p] + sum_{p,q} Lmbdac_{pq} M_Lc_full[p,q].
    D_sp:      (nqspo,) hybridization strengths in original quasi-orbital basis (real)
    Lmbdac:    (nqspo,nqspo) bath energy matrix in original quasi-orbital basis (real)
    M_Lc_full: (nqspo,nqspo,dim,dim) bath-bath operator matrix

    No fix_gauge is applied — D_sp and Lmbdac are used directly in the original basis
    in which M_D and M_Lc_full were precomputed.
    """
    H = H_const.copy()
    H += np.einsum('p,pij->ij', np.real(D_sp), M_D)
    H += np.einsum('pq,pqij->ij', np.real(Lmbdac), M_Lc_full)
    return H


# ============================================================
# fdaggerc_sp from wavefunction  [Eq.20 constraint]
# ============================================================
def compute_fdaggerc_sp(Phi, params):
    """Compute fdaggerc in spatial quasi-orbital space from |Phi>."""
    op_bp = params['op_bp']
    n_phys = params['n_phys']
    n_bath = params['n_bath']
    nqspo = params['nqspo']
    Phi_c = np.conj(Phi)

    # spin-space fdaggerc
    fdc = np.array([[np.real(Phi_c @ (op_bp[i, j] @ Phi))
                     for j in range(n_phys)]
                    for i in range(n_bath)])  # (n_bath, n_phys)

    # spin-symmetrize → spatial (nqspo, 1)
    fdc_sp = np.array([[0.5 * (fdc[2 * i, 0] + fdc[2 * i + 1, 1])]
                       for i in range(nqspo)])
    return fdc_sp  # (nqspo, 1)


# ============================================================
# Fast Lmbdac: eigh(Delta) once instead of n_basis times
# ============================================================
def calc_Lmbdac_fast(Delta, D, R, H_list, H_list_arr):
    """
    Optimized calc_Lmbdac for Lambda=0 (TD case).
    Calls eigh(Delta) once; vectorizes over all H_list basis matrices.
    Lambda=0 gauge: no Route B correction needed.
    """
    from scipy.linalg import eigh as scipy_eigh
    nqspo = Delta.shape[0]
    DxR = D @ R.T  # (nqspo, nqspo)

    evals, evecs = scipy_eigh(np.real(Delta))
    evals = np.clip(evals, 1e-14, 1.0 - 1e-14)

    # Loewner matrix for denRm1 = sqrt(x(1-x))
    f_vals = np.sqrt(evals * (1.0 - evals))
    df_vals = (0.5 - evals) / f_vals
    L = np.zeros((nqspo, nqspo))
    for i in range(nqspo):
        for j in range(nqspo):
            if i == j or abs(evals[i] - evals[j]) < 1e-12:
                L[i, j] = df_vals[i]
            else:
                L[i, j] = (f_vals[i] - f_vals[j]) / (evals[i] - evals[j])

    # M = evecs.T @ DxR @ evecs  (once)
    M = evecs.T @ DxR @ evecs  # (nqspo, nqspo)

    # Batch transform all H_list_arr bases: H_bar[s] = evecs.T @ H_list_arr[s] @ evecs
    H_bar = np.einsum('ij,sjk,kl->sil', evecs.T, H_list_arr, evecs)

    # traces[s] = trace(M @ (L * H_bar[s]))
    lmbdac = -2.0 * np.real(np.einsum('ij,sji->s', M, L[np.newaxis] * H_bar))

    return cr.realHcombination(lmbdac, H_list)


# ============================================================
# Clamp Delta eigenvalues for numerical stability
# ============================================================
def safe_Delta(Delta):
    evals, evecs = np.linalg.eigh(np.real(Delta))
    evals = np.clip(evals, 1e-6, 1.0 - 1e-6)
    return evecs @ np.diag(evals) @ evecs.T


# ============================================================
# Compute R, D, Lmbdac from (Phi, n_omega)
# ============================================================
def compute_RDLc(Phi, n_omega, params, ga_obj):
    """
    Given current (Phi, n_omega), compute R, D, Lmbdac.
    Sets ga_obj.D and ga_obj.R as side-effect (needed by calc_Lmbdac).
    """
    omega_arr = params['omega_arr']
    weights = params['weights']
    rho_arr = params['rho_arr']
    nqspo = params['nqspo']

    w_rho = weights * rho_arr                                        # (N_freq,)
    Delta = safe_Delta(np.einsum('f,fab->ab', w_rho, np.real(n_omega)))
    B_mat = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr, n_omega))

    fdc_sp = compute_fdaggerc_sp(Phi, params)                        # (nqspo,1)

    # [Delta(1-Delta)]^{-1/2} — compute once, reuse for R and D
    denR_mat = cr.funcMat(Delta, cr.denR)                            # (nqspo,nqspo)

    # R from Eq.(20): R = [Delta(1-Delta)]^{-1/2} @ fdaggerc_sp
    R = (fdc_sp.T @ denR_mat).T                                      # (nqspo,1)

    # D from Eq.(18): D = [Delta(1-Delta)]^{-1/2} @ (R^T @ B_mat)^T
    Left = (np.real(R[:, 0]) @ B_mat).reshape(nqspo, 1)
    D = denR_mat @ Left                                              # (nqspo,1)

    # Lambda^c: Lc = Lc_fast - Λ  (Route B; Λ fixed at static value)
    ga_obj.D = D
    ga_obj.R = R
    Lmbdac = calc_Lmbdac_fast(Delta, D, R,
                               params['H_list'], params['H_list_arr'])
    Lmbda_npz = params.get('Lmbda_npz')
    if Lmbda_npz is not None:
        Lmbdac = Lmbdac - np.real(Lmbda_npz)

    return R, D, Lmbdac, Delta


# ============================================================
# ODE right-hand side  F(Y)
# ============================================================
def compute_derivatives_full(t, y, params, ga_obj):
    dim_Phi = params['dim_Phi']
    N_freq = params['N_freq']
    nqspo = params['nqspo']
    omega_arr = params['omega_arr']
    M_D = params['M_D']
    M_Lc_full = params['M_Lc_full']
    H_const = params['H_const']

    Phi, n_omega = unpack_state(y, dim_Phi, N_freq, nqspo)
    Phi /= np.linalg.norm(Phi)

    # Solve constraints → get R, D, Lc
    R, D, Lmbdac, Delta = compute_RDLc(Phi, n_omega, params, ga_obj)

    # Eq.(16): d|Phi>/dt = -i H_emb |Phi>
    # Use D and Lmbdac directly in the original quasi-orbital basis (no fix_gauge).
    # fix_gauge would rotate D into the Lmbdac eigenbasis, but M_D and M_Lc_full are
    # precomputed in the original basis — mixing them would produce incorrect H_emb.
    D_sp = np.real(D[:, 0])
    H_emb = build_H_emb_fast(D_sp, np.real(Lmbdac), H_const, M_D, M_Lc_full)
    dPhi_dt = -1j * (H_emb @ Phi)

    # Eq.(17): dn/dt = -iω[n, RR†] + i[Λ, n]  (Route B)
    # [n, RR†]_{ab} = sum_c (n_{ac} R_c R_b* - R_a R_c* n_{cb})
    R_vec = R[:, 0]                         # (nqspo,)
    R_conj = np.conj(R_vec)
    v = np.einsum('fab,b->fa', n_omega, R_vec)    # n·R  (no conj)
    u = np.einsum('a,fab->fb', R_conj, n_omega)   # R*·n
    dn_dt = -1j * omega_arr[:, None, None] * (
        np.einsum('fa,b->fab', v, R_conj) -        # (n·R)⊗R*
        np.einsum('a,fb->fab', R_vec, u)            # R⊗(R*·n)
    )
    # +i[Λ, n] = i(Λ·n - n·Λ)  — variational term from Route B
    Lmbda_npz = params.get('Lmbda_npz')
    if Lmbda_npz is not None:
        L = np.real(Lmbda_npz)
        dn_dt += 1j * (L[None] @ n_omega - n_omega @ L[None])

    return pack_state(dPhi_dt, dn_dt)


# ============================================================
# 縮退 H_emb で F2 を最小化する Φ を選ぶ（run_full_simulation SC ループ用）
# ============================================================
def _best_phi_from_degenerate(H_emb_dense, Delta_i, op_bb, nqspo,
                               tol_degen=0.05, n_tries=10):
    """
    密 H_emb の縮退部分空間で ||⟨f†f⟩ − Δ|| を最小化する Φ を返す。
    縮退なければ通常の基底状態を返す。
    """
    from scipy.optimize import minimize

    evals, evecs = np.linalg.eigh(H_emb_dense)
    E0         = evals[0]
    degen_mask = (evals - E0) < tol_degen
    V          = evecs[:, degen_mask]   # (dim_Phi, n_deg)
    n_deg      = V.shape[1]

    if n_deg <= 1:
        phi = evecs[:, 0]
        return phi / np.linalg.norm(phi)

    # M[a,b] = 0.5*(V.T @ op_bb[2a,2b] @ V + V.T @ op_bb[2a+1,2b+1] @ V)
    M = np.zeros((nqspo, nqspo, n_deg, n_deg))
    for a in range(nqspo):
        for b in range(nqspo):
            M[a, b] = 0.5 * (V.T @ op_bb[2*a, 2*b] @ V +
                              V.T @ op_bb[2*a+1, 2*b+1] @ V)

    def residual(x):
        norm = np.linalg.norm(x)
        c    = x / norm if norm > 1e-300 else x
        ffd  = np.einsum('abkl,k,l->ab', M, c, c)
        return float(np.linalg.norm(ffd - Delta_i) ** 2)

    rng      = np.random.default_rng(42)
    best_val = np.inf
    best_c   = np.zeros(n_deg); best_c[0] = 1.0

    for _ in range(n_tries):
        x0  = rng.standard_normal(n_deg)
        x0 /= np.linalg.norm(x0)
        res  = minimize(residual, x0, method='COBYLA',
                        options={'maxiter': 1000, 'rhobeg': 0.3})
        if res.fun < best_val:
            best_val = res.fun
            best_c   = res.x / (np.linalg.norm(res.x) + 1e-300)

    phi  = V @ best_c
    return phi / np.linalg.norm(phi)


# ============================================================
# Run full simulation
# ============================================================
def run_full_simulation(ga_obj, U_final, t_max, dt, N_freq=100):
    params = prepare_full_params(ga_obj, U_final, N_freq)
    omega_arr = params['omega_arr']
    nqspo = ga_obj.nqspo

    # Self-consistently solve for (Phi_0, R_0, n0_omega) in the original basis.
    # Key insight (STATUS.md "正しい解法"): compute D from R_cur directly (Step2c),
    # without going through Phi.  This avoids the fix_gauge basis mismatch that
    # prevented convergence in earlier attempts.
    # Initial R_cur from imp_solver (fix_gauge ordering); convert to npz ordering
    # so the entire SC loop operates in a single consistent basis (npz).
    Delta_s = safe_Delta(np.real(ga_obj.imp_solver.ffdagger[:nqspo, :nqspo]))
    fdc_s   = np.real(ga_obj.imp_solver.fdaggerc)
    R_fg    = cr.funcMat(Delta_s, cr.denR) @ fdc_s  # fix_gauge ordering
    P       = params['P_npz2imp']
    R_cur   = P.T @ R_fg                             # convert to npz ordering

    w_rho = params['weights'] * params['rho_arr']
    Phi_0 = None
    n0_iter = None
    evals_i = None

    print("  Self-consistently solving for (Phi_0, R_0, n0_omega) in original basis...")
    print(f"  (初期状態 SC: Route A 式, Λ=0 固定)")
    for sc_iter in range(100):
        # Step2a: n(ω) = f(ω RR†)  — Route A: Λ=0 gauge
        RopRt_cur = R_cur @ R_cur.T
        n0_iter = np.array([cr.calc_C(RopRt_cur * omega, T=ga_obj.T)
                            for omega in omega_arr])

        # Step2b: Delta, B_mat
        Delta_i = safe_Delta(np.einsum('f,fab->ab', w_rho, np.real(n0_iter)))
        B_mat_i = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr, n0_iter))
        denR_i  = cr.funcMat(Delta_i, cr.denR)

        # Step2c: D from R_cur (no Phi) — avoids basis mismatch
        Left_i = (R_cur[:, 0] @ B_mat_i).reshape(nqspo, 1)
        D_i    = denR_i @ Left_i

        # Step2d: Lc = Lc_fast  (Λ=0 なので補正なし)
        Lc_i = calc_Lmbdac_fast(Delta_i, D_i, R_cur,
                                 params['H_list'], params['H_list_arr'])

        # Step2e-f: H_emb → ground state (use U_initial H_const for initial state)
        H_emb_i = build_H_emb_fast(np.real(D_i[:, 0]), np.real(Lc_i),
                                    params['H_const_init'], params['M_D'], params['M_Lc_full'])
        Phi_i = _best_phi_from_degenerate(H_emb_i, Delta_i, params['op_bb'], nqspo)

        # Step2g-h: R_new from Phi_i
        fdc_new = compute_fdaggerc_sp(Phi_i, params)
        R_new   = (fdc_new.T @ denR_i).T  # (nqspo,1)

        dR    = np.linalg.norm(R_new - R_cur)
        R_cur = R_new
        Phi_0 = Phi_i
        if dR < 1e-10:
            print(f"  Converged in {sc_iter+1} iterations (dR={dR:.2e})")
            break
    else:
        print(f"  Warning: SC loop did not converge after 100 iterations (dR={dR:.2e})")

    n0_omega = n0_iter  # final n(ω) from converged R_cur
    # Update ga_obj so ODE bookkeeping starts from consistent values
    ga_obj.D = D_i
    ga_obj.R = R_cur
    evals_final = np.linalg.eigvalsh(H_emb_i)
    print(f"  E_0 = {evals_final[0]:.6f}  (gap = {evals_final[1]-evals_final[0]:.4f})")

    y0 = pack_state(Phi_0, n0_omega)
    t_eval = np.arange(0.0, t_max, dt)

    print(f"  State size: {len(y0)} reals")
    print(f"  Integrating from t=0 to t={t_max}...")

    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        compute_derivatives_full,
        [0.0, t_max],
        y0,
        t_eval=t_eval,
        args=(params, ga_obj),
        method='RK45',
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )
    print(f"  Integration done. {sol.nfev} function evaluations.")

    # ---- Post-analysis ----
    print("  Post-processing...")
    dim_Phi = params['dim_Phi']
    N_freq_p = params['N_freq']
    op_docc = params['op_docc']
    op_bb   = params['op_bb']
    op_n_orb_list = params['op_n_orb_list']
    n_orb_count = len(op_n_orb_list)
    weights = params['weights']
    rho_arr = params['rho_arr']
    w_rho = weights * rho_arr
    omega_arr_p = params['omega_arr']
    M_D = params['M_D']
    M_Lc_full = params['M_Lc_full']
    H_const = params['H_const']
    Lmbda_npz_p = params['Lmbda_npz']
    n_bath = params['n_bath']

    docc_list     = []
    E_emb_list    = []   # <Phi|H_emb|Phi>
    E_qp_list     = []   # 2 Tr[RR† B_mat]
    E_lmbda_list  = []   # Tr[Λ Δ]  (Route B 補正項)
    E_B_list      = []   # E_qp + E_lmbda + E_emb  (Route B 保存量候補)
    E_phys_list   = []   # E_qp + U_final * d       (物理 Hubbard エネルギー)
    F2_list       = []   # ||<f†f>_Phi - Δ_n||       (拘束条件違反)
    TrDelta_list  = []   # Tr[Δ]                     (占有数チェック)
    Z_list        = []
    n_orb_lists   = [[] for _ in range(n_orb_count)]
    eig_Lc_list   = []
    sqrtDtD_list  = []

    for k in range(len(sol.t)):
        Phi_t, n_omega_t = unpack_state(sol.y[:, k], dim_Phi, N_freq_p, nqspo)
        Phi_t /= np.linalg.norm(Phi_t)
        Phi_c = np.conj(Phi_t)

        # Double occupancy
        d_t = float(np.real(np.dot(Phi_c, op_docc @ Phi_t)))
        docc_list.append(d_t)

        # Orbital occupancies
        for i in range(n_orb_count):
            n_orb_lists[i].append(float(np.real(np.dot(Phi_c, op_n_orb_list[i] @ Phi_t))))

        # Constraints → R, D, Lc, Δ
        R_t, D_t, Lmbdac_t, Delta_t = compute_RDLc(Phi_t, n_omega_t, params, ga_obj)

        # Z(t): Route A uses Λ=0, so calc_Z with zero Lmbda → Z = Tr(RR†) limit
        R_t_fg = params['P_npz2imp'] @ R_t
        Lmbda_zero = np.zeros((nqspo, nqspo))
        Z_list.append(ga_obj.calc_Z(Lmbda_zero, R_t_fg))

        # eig(Λc), √(D†D)
        eig_Lc_list.append(np.sort(np.linalg.eigvalsh(Lmbdac_t)))
        sqrtDtD_list.append(float(np.sqrt(np.real(np.dot(D_t[:, 0], D_t[:, 0])))))

        # E_emb = <Phi|H_emb|Phi>
        D_sp_t = np.real(D_t[:, 0])
        H_emb_t = build_H_emb_fast(D_sp_t, np.real(Lmbdac_t), H_const, M_D, M_Lc_full)
        E_emb_t = float(np.real(np.dot(Phi_c, H_emb_t @ Phi_t)))
        E_emb_list.append(E_emb_t)

        # B_mat = ∫dω ρ(ω) ω n(ω)
        B_mat_t = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr_p, n_omega_t))
        R_vec_t = np.real(R_t[:, 0])

        # E_qp = 2 Tr[RR† B_mat]
        E_qp_t = 2.0 * float(R_vec_t @ B_mat_t @ R_vec_t)
        E_qp_list.append(E_qp_t)

        # Tr[Λ Δ]
        E_lmbda_t = float(np.trace(np.real(Lmbda_npz_p) @ np.real(Delta_t)))
        E_lmbda_list.append(E_lmbda_t)

        # Route B 保存量候補: E_B = E_qp + Tr[ΛΔ] + E_emb
        E_B_list.append(E_qp_t + E_lmbda_t + E_emb_t)

        # 物理 Hubbard エネルギー: E_phys = E_qp + U_final * d
        E_phys_list.append(E_qp_t + U_final * d_t)

        # Tr[Δ]（粒子数チェック、半充填なら ~1）
        TrDelta_list.append(float(np.trace(np.real(Delta_t))))

        # F2 = ||<f†f>_Phi - Δ_n||  (拘束条件 Eq.21 の違反)
        # <f†f>_ab を Phi から計算（スピン平均）
        ffdagger_phi = np.array([[0.5 * (np.real(Phi_c @ (op_bb[2*i, 2*j] @ Phi_t)) +
                                         np.real(Phi_c @ (op_bb[2*i+1, 2*j+1] @ Phi_t)))
                                  for j in range(nqspo)]
                                 for i in range(nqspo)])
        F2_list.append(float(np.linalg.norm(ffdagger_phi - np.real(Delta_t))))

    results = {
        't':          sol.t,
        'docc':       np.array(docc_list),
        'E_emb':      np.array(E_emb_list),    # <Phi|H_emb|Phi>
        'E_qp':       np.array(E_qp_list),     # 2 Tr[RR†B]
        'E_lmbda':    np.array(E_lmbda_list),  # Tr[ΛΔ]
        'E_B':        np.array(E_B_list),      # Route B 保存量候補
        'E_phys':     np.array(E_phys_list),   # 物理エネルギー
        'F2':         np.array(F2_list),       # 拘束条件違反
        'TrDelta':    np.array(TrDelta_list),  # Tr[Δ]
        'Z':          np.array(Z_list),
        'n_orb':      np.array(n_orb_lists),
        'eig_Lc':     np.array(eig_Lc_list),
        'sqrtDtD':    np.array(sqrtDtD_list),
        # 後方互換
        'E_tot':      np.array(E_emb_list),
    }
    return results


# ============================================================
# 断熱的初期化: 小さい U での収束問題を回避する
# ============================================================
def solve_static_adiabatic(U_target, nghost=4, nphysorb=2, n=0.5, T=0.003,
                           U_ref=0.5, n_steps=10):
    """
    Route A (Λ=0 ゲージ) の静的 gGA を断熱的に解く。

    U_ref から U_target まで n_steps 段階で U を変えながら、
    前の解を初期値として使い回す。各ステップで optimize_selfc_routeA を呼ぶ。

    Returns: 収束した GA オブジェクト (Lmbda=0 保証済み)
    """
    from gga_static_solver import GA

    calc_nqspo = (nphysorb + nghost) // 2

    if U_target >= U_ref:
        print(f"  [adiabatic] U_target={U_target} >= U_ref={U_ref}: 直接解きます")
        ga = GA(U=U_target, nghost=nghost, nphysorb=nphysorb, n=n, T=T, eks=-99)
        rinit = np.ones(calc_nqspo) / np.sqrt(calc_nqspo)
        ga.optimize_selfc_routeA(rinit=rinit, muinit=0.0)
        return ga

    U_vals = np.exp(np.linspace(np.log(U_ref), np.log(U_target), n_steps + 1))

    rinit  = np.ones(calc_nqspo) / np.sqrt(calc_nqspo)
    muinit = 0.0
    ga_prev = None

    for step, U in enumerate(U_vals):
        print(f"  [adiabatic] step {step+1}/{len(U_vals)}: U={U:.4f}")
        ga = GA(U=float(U), nghost=nghost, nphysorb=nphysorb, n=n, T=T, eks=-99)
        if ga_prev is not None:
            ga.mu  = ga_prev.mu
            rinit  = ga_prev.R[:, 0]
            muinit = float(ga_prev.mu)
        ga.optimize_selfc_routeA(rinit=rinit, muinit=muinit)
        print(f"           → Z={ga.Z:.5f}, d={float(np.real(ga.imp_solver.docc)):.5f}")
        ga_prev = ga

    return ga_prev

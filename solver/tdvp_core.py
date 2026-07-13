"""
tdvp_core.py

td_gGA_solver.py が使う基礎ヘルパー関数群:
  - setup_frequency_grid: バス周波数グリッド（Gauss-Legendre × 半円DOS）
  - pack_state/unpack_state: ODE状態ベクトルと (Φ, n(ω)) の相互変換
  - prepare_full_params: 演算子npzを読み込み密行列paramsを構築（B<=5想定）
  - compute_fdaggerc_sp: |Φ⟩から⟨f†c⟩を計算

これらの関数はもともと Route A（このファイル自身の独自TD時間発展ドライバ、
Λ=0固定・規約バグあり）のために書かれた。Route A のドライバ本体は使われて
おらず、正しい実装は td_gGA_solver.py にある。ドライバ本体は削除済み
（検証時の経緯は POSTMORTEM_2026-07-07.md, derivation_F2_conservation_proof.md 参照）。
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

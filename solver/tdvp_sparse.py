"""
tdvp_sparse.py

【このファイルの現在の役割】
このファイル自身の時間発展ドライバ（下の "Route C" 一式）は使われていない
（B=3以上でエネルギー非保存など既知の問題あり）。実際に使われているのは
prepare_full_params, pack_state/unpack_state, compute_ffdagger_sp,
compute_fdaggerc_sp, solve_lambda_from_F2 などのヘルパー関数群（疎行列対応、
B=5,7 向け）で、これらは正しい実装である td_gGA_solver.py から import
されて使われている。以下のドキュメントは元の（未使用の）ドライバの説明として残してある。

--- 以下、旧ドキュメント ---
TD-gGA Route C: Λ(t) is determined dynamically at each step from F2=0.

Key differences from Route A (Λ=0 fixed):
  - Λ(t) is solved at every ODE evaluation from the constraint
      Δ(Λ) ≡ ∫ρ(ω) f(ω RR† + Λ) dω  =  ⟨Φ|f†f|Φ⟩
  - n ODE includes the Λ(t) commutator term:
      dn/dt = -iω [n, RR†] + i [Λ, n]
  - H_emb uses  Λ^c = Λ^c_fast − Λ(t)  (Λ is non-zero)
  - Initial state from full gGA equilibrium (optimize_selfc, Λ_eq ≠ 0)
    → n_0(ω) = f(ω RR†_0 + Λ_eq) ≠ f(ω RR†_0) alone
    → [RR†_0, n_0] ≠ 0 already at t=0 → genuine B>1 evolution

Route A and C share: prepare_full_params, pack/unpack, build_H_emb_fast,
calc_Lmbdac_fast, safe_Delta, compute_fdaggerc_sp.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from math import comb
from numpy.polynomial.legendre import leggauss
import convenience_routines as cr

# Re-use pure utilities from Route A
from tdvp_core import (
    setup_frequency_grid,
    pack_state,
    unpack_state,
    prepare_full_params,
    build_H_emb_fast,
    calc_Lmbdac_fast,
    safe_Delta,
    compute_fdaggerc_sp,
    _best_phi_from_degenerate,
)


# ============================================================
# Bath-bath density matrix from |Φ⟩
# ============================================================
def compute_ffdagger_sp(Phi, params):
    """
    Compute the electron density matrix ⟨Φ|f†_i f_j|Φ⟩ (spatial, spin-averaged).
    Returns (nqspo, nqspo) real matrix.

    Note: op_bb[i,j] = f_j f†_i (lreverse=True), so
          ⟨op_bb[i,j]⟩ = δ_{ij} - ⟨f†_i f_j⟩  (hole density).
    We return the electron density: I - hole.
    """
    op_bb = params['op_bb']
    nqspo = params['nqspo']
    Phi_c = np.conj(Phi)
    hole = np.array([[0.5 * (np.real(Phi_c @ (op_bb[2 * i, 2 * j] @ Phi)) +
                             np.real(Phi_c @ (op_bb[2 * i + 1, 2 * j + 1] @ Phi)))
                      for j in range(nqspo)]
                     for i in range(nqspo)])
    return np.eye(nqspo) - hole  # ⟨f†f⟩ = I - hole


# ============================================================
# Λ solver: algebraic F2(Λ) = 0 (used for initialization only)
# ============================================================
def solve_lambda_from_F2(ffd_target, R, Lmbda_init,
                          omega_arr, w_rho, T,
                          H_list, tol=1e-8, maxiter=100):
    """
    Find Λ such that ∫ρ(ω) f(ω RR† + Λ) dω = ffd_target.
    Used for initialization (t=0) and post-processing only.
    During time evolution, solve_lambda_differential is used instead.
    """
    RRt = R @ R.T

    def residual(lmbda_vec):
        Lmbda = cr.realHcombination(lmbda_vec, H_list)
        Delta_L = np.einsum('f,fab->ab', w_rho,
                            np.array([cr.calc_C(RRt * om + Lmbda, T=T)
                                      for om in omega_arr]))
        F2 = Delta_L - ffd_target
        return cr.inverse_realHcombination(F2, H_list)

    lmbda0 = cr.inverse_realHcombination(Lmbda_init, H_list)
    res = least_squares(residual, lmbda0, method='lm',
                        ftol=tol, xtol=tol, gtol=tol, max_nfev=maxiter)
    return cr.realHcombination(res.x, H_list)


# ============================================================
# Λ solver: differential F2 constraint (proper Route C)
# ============================================================
def solve_lambda_differential(Phi, H_emb_Phi, B_mat, R_vec, ffd, params):
    """
    Solve for Λ from the differential F2=0 constraint:
        [Λ, n_el_phi] = ⟨[H_emb, f†f]⟩ + [B_mat, RR†]

    This ensures d/dt(∫ρ n_om) = d/dt n_el_phi, so F2=0 is preserved.
    Off-diagonal elements of Λ (in eigenbasis of n_el_phi) are uniquely determined.
    Diagonal elements are fixed to zero (gauge choice in n_el_phi eigenbasis).

    H_emb_Phi = H_emb @ Phi (precomputed to avoid recomputation).
    H_emb must be real symmetric (standard for gGA embedding Hamiltonian).
    """
    nqspo = params['nqspo']
    op_bb = params['op_bb']

    Phi_c = np.conj(Phi)
    H_emb_Phi_c = np.conj(H_emb_Phi)   # H_emb real ⟹ H_emb Phi* = (H_emb Phi)*

    # ---- Step A: compute ⟨[H_emb, f†_{i}f_{j}]⟩ via ⟨[op_bb[s,i,j], H_emb]⟩ ----
    # op_bb[2i,2j] = f_{j,↑}f†_{i,↑}  (lreverse=True)
    # f†_{i}f_{j} = δ_{ij} - op_bb[2i,2j]  ⟹  [H_emb, f†f]_{ij} = -[H_emb, op_bb]_{ij}
    # ⟨[op, H_emb]⟩ = ⟨op H_emb⟩ - ⟨H_emb op⟩
    #                = Phi_c.(op @ H_emb_Phi) - H_emb_Phi_c.(op @ Phi)
    Hn_comm = np.zeros((nqspo, nqspo))
    for i in range(nqspo):
        for j in range(nqspo):
            comm_s = 0.0
            for s in range(2):
                op = op_bb[2 * i + s, 2 * j + s]
                op_Phi = op @ Phi
                # ⟨[op, H_emb]⟩ = ⟨op H_emb⟩ - ⟨H_emb op⟩
                c = (np.real(Phi_c @ (op @ H_emb_Phi))
                     - np.real(H_emb_Phi_c @ op_Phi))
                comm_s += c
            # ⟨[H_emb, f†_{i}f_{j}]⟩ = -spin-avg ⟨[op_bb[s,i,j], H_emb]⟩
            Hn_comm[i, j] = -0.5 * comm_s

    # ---- Step B: RHS = ⟨[H_emb, f†f]⟩ + [B_mat, RR†] ----
    RRt = R_vec[:, None] * R_vec[None, :]
    RHS = Hn_comm + (B_mat @ RRt - RRt @ B_mat)

    # ---- Step C: solve [Λ, ffd] = RHS in eigenbasis of ffd ----
    eig_n, U_n = np.linalg.eigh(ffd)           # eig_n ascending
    RHS_rot = U_n.T @ RHS @ U_n                 # rotate to diagonal basis

    # Diagonal = 0 (gauge choice): makes Λ a deterministic function of (Φ, n_ω)
    Lmbda_rot = np.zeros((nqspo, nqspo))
    for i in range(nqspo):
        for j in range(nqspo):
            if i != j:
                denom = eig_n[j] - eig_n[i]
                if abs(denom) > 1e-10:
                    Lmbda_rot[i, j] = RHS_rot[i, j] / denom
                # else: nearly degenerate — leave as zero

    return U_n @ Lmbda_rot @ U_n.T


# ============================================================
# ODE right-hand side for Route C
# ============================================================
def compute_derivatives_routeC(t, y, params, ga_obj):
    """
    Route C ODE rhs.  Λ is computed deterministically at each call
    (diagonal-zero gauge in n_el_phi eigenbasis) — no shared mutable state.
    """
    dim_Phi = params['dim_Phi']
    N_freq  = params['N_freq']
    nqspo   = params['nqspo']
    omega_arr   = params['omega_arr']
    weights     = params['weights']
    rho_arr     = params['rho_arr']
    M_D         = params['M_D']
    M_Lc_full   = params['M_Lc_full']
    H_const     = params['H_const']
    H_list      = params['H_list']
    H_list_arr  = params['H_list_arr']

    w_rho = weights * rho_arr

    Phi, n_omega = unpack_state(y, dim_Phi, N_freq, nqspo)
    Phi /= np.linalg.norm(Phi)

    # --- Step 1: B_mat from n(ω) ODE  [physically correct bath memory] ---
    B_mat = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr, n_omega))

    # --- Step 2: Δ_n = n_el_phi ---
    ffd     = compute_ffdagger_sp(Phi, params)     # ⟨f†f⟩ from Φ
    Delta_n = safe_Delta(ffd)

    # --- Step 3: R from ⟨f†c⟩ ---
    denR_mat = cr.funcMat(Delta_n, cr.denR)
    fdc      = compute_fdaggerc_sp(Phi, params)
    R        = (fdc.T @ denR_mat).T
    R_vec    = R[:, 0]

    # --- Step 4: D from Eq.(18) ---
    Left = (R_vec @ B_mat).reshape(nqspo, 1)
    D    = denR_mat @ Left

    # --- Steps 5-7: Self-consistent Λ iteration ---
    # Λ satisfies [Λ, n_el] = ⟨[H_emb(Λ), f†f]⟩ + [B,RR†], a fixed-point equation.
    # Warm-start: use params['Lmbda_warm'] if available (set externally between sequential
    # integration steps), otherwise fall back to Lmbda_eq.  We do NOT update the warm-start
    # inside the ODE rhs because explicit Runge-Kutta solvers evaluate the rhs at intermediate
    # stages and may retry with smaller step sizes — updating mid-step contaminates the
    # warm-start with rejected-step Λ values and degrades metallic-quench accuracy.
    n_lambda_iter = params.get('n_lambda_iter', 3)
    Lmbda = params.get('Lmbda_warm', params.get('Lmbda_eq', np.zeros((nqspo, nqspo)))).copy()
    for _ in range(n_lambda_iter):
        Lmbdac_iter = calc_Lmbdac_fast(Delta_n, D, R, H_list, H_list_arr) - Lmbda
        H_emb_iter  = build_H_emb_fast(np.real(D[:, 0]), np.real(Lmbdac_iter),
                                        H_const, M_D, M_Lc_full)
        Lmbda = solve_lambda_differential(Phi, H_emb_iter @ Phi, B_mat, R_vec, ffd, params)

    Lmbdac = calc_Lmbdac_fast(Delta_n, D, R, H_list, H_list_arr) - Lmbda
    H_emb  = build_H_emb_fast(np.real(D[:, 0]), np.real(Lmbdac), H_const, M_D, M_Lc_full)
    dPhi_dt = -1j * (H_emb @ Phi)

    # --- Step 8: n(ω) ODE  dn/dt = -iω[n, RR†] + i[Λ, n] + γ(n_el − ∑ρn) ---
    # [n, RR†]_{ab} = sum_c (n_{ac} R_c R_b* - R_a R_c* n_{cb})
    R_conj = np.conj(R_vec)
    v = np.einsum('fab,b->fa', n_omega, R_vec)    # n·R  (no conj)
    u = np.einsum('a,fab->fb', R_conj, n_omega)   # R*·n
    dn_dt = -1j * omega_arr[:, None, None] * (
        np.einsum('fa,b->fab', v, R_conj) -        # (n·R)⊗R*
        np.einsum('a,fb->fab', R_vec, u)            # R⊗(R*·n)
    )
    L = np.real(Lmbda)
    dn_dt += 1j * (L[None] @ n_omega - n_omega @ L[None])

    # F2制約安定化: ∑ρ n(ω) → n_el を促すペナルティ項（Baumgarte-Stabier法）
    # F2=0が保たれていれば補正項≈0; F2が蓄積したら元に戻す
    gamma_stab = params.get('gamma_stab', 10.0)
    Delta_n_bath = np.einsum('f,fab->ab', w_rho, np.real(n_omega))
    dn_dt += gamma_stab * (ffd - Delta_n_bath)[None, :, :]

    return pack_state(dPhi_dt, dn_dt)


# ============================================================
# Run Route C simulation
# ============================================================
def run_full_simulation_routeC(ga_obj, U_final, t_max, dt, N_freq=100, extra_params=None):
    """
    Route C quench simulation.

    ga_obj must be from optimize_selfc (full gGA, Λ_eq ≠ 0).
    Initial state is taken directly from ga_obj equilibrium — no SC loop.
    """
    from scipy.integrate import solve_ivp

    params  = prepare_full_params(ga_obj, U_final, N_freq)
    if extra_params:
        params.update(extra_params)
    nqspo   = ga_obj.nqspo
    omega_arr = params['omega_arr']
    weights   = params['weights']
    rho_arr   = params['rho_arr']
    w_rho     = weights * rho_arr
    P         = params['P_npz2imp']

    # ---- Initial |Φ_0⟩ from optimize_selfc ground state ----
    # ed.eig_vec is in the original (npz) Fock space — same basis as op_bb, op_bp.
    # (fix_gauge only transforms D and Lmbdac coefficients; the Fock basis does not change.)
    Phi_0 = params['Phi_0'].copy()
    print(f"  初期 |Φ_0⟩ ノルム: {np.linalg.norm(Phi_0):.6f}")

    # ---- Initial density matrices in npz basis directly from Phi_0 ----
    # At t=0, n_om_0 = f(ωRR†+Λ_eq), so Δ_n_0 = n_el_0 (F2=0 at t=0).
    # We compute n_el_0 from Phi_0 (= Δ_n_0), then use it for R0 (TDVP F1 condition).
    n_el_0 = compute_ffdagger_sp(Phi_0, params)    # ⟨f†f⟩ from Φ_0
    fdc_0  = compute_fdaggerc_sp(Phi_0, params)    # ⟨f†c⟩ from Φ_0
    R0_npz = (fdc_0.T @ cr.funcMat(safe_Delta(n_el_0), cr.denR)).T
    print(f"  R0 (npz): {np.round(R0_npz[:, 0], 4)}")
    print(f"  n_el_0 対角: {np.round(np.diag(n_el_0), 4)}")

    # ---- Compute Λ_eq in npz basis via eigenvector matching ----
    # n_el_0 diagonalizes in the fix_gauge single-particle axes.
    # Those same axes diagonalize Λ_eq, so:
    #   n_el_0 = U_npz @ diag(n_eigs) @ U_npz.T
    #   Λ_eq_npz = U_npz @ diag(Λ_eigs_matched) @ U_npz.T
    # Pairing: n_el ascending (≈0, 0.5, 1) ↔ Λ descending (max→min).
    eig_n, U_npz = np.linalg.eigh(n_el_0)        # eig_n ascending: [~0, ~0.5, ~1]
    eig_Lmbda = np.linalg.eigvalsh(np.real(ga_obj.Lmbda))  # ascending: [~-0.67, 0, ~0.67]
    # n_el=0 → Λ=max_positive (empty mode); n_el=1 → Λ=min_negative (full mode)
    Lmbda_guess = U_npz @ np.diag(eig_Lmbda[::-1]) @ U_npz.T

    # Refine with F2 solve (should converge quickly from this close initial guess)
    Lmbda_eq_npz = solve_lambda_from_F2(
        n_el_0, R0_npz, Lmbda_guess,
        omega_arr, w_rho, ga_obj.T, params['H_list'],
        tol=1e-10, maxiter=500,
    )
    print(f"  Λ_eq 固有値 (npz basis): {np.round(np.linalg.eigvalsh(Lmbda_eq_npz), 4)}")
    params['Lmbda_eq'] = Lmbda_eq_npz  # warm-start for ODE Λ iteration

    # ---- Initial n_0(ω) = f(ω RR† + Λ_eq) ----
    RRt0     = R0_npz @ R0_npz.T
    n0_omega = np.array([cr.calc_C(RRt0 * om + Lmbda_eq_npz, T=ga_obj.T)
                         for om in omega_arr])

    # Check initial commutator [RR†, n0] to verify non-frozen n
    comm_norms = [np.linalg.norm(RRt0 @ n0_omega[k] - n0_omega[k] @ RRt0)
                  for k in range(N_freq)]
    print(f"  |[RR†, n0]| 平均: {np.mean(comm_norms):.3e}  (Route A では 0)")

    # ---- Initial F2 check (should be ~0 by construction) ----
    Delta0  = np.einsum('f,fab->ab', w_rho, np.real(n0_omega))
    F2_init = np.linalg.norm(n_el_0 - Delta0)
    print(f"  初期 F2 = {F2_init:.3e}")

    # ---- ODE ----
    import time as _time
    y0 = pack_state(Phi_0, n0_omega)
    t_eval = np.arange(0.0, t_max, dt)

    print(f"  State size: {len(y0)} reals")
    print(f"  Integrating from t=0 to t={t_max}...")

    _call_count = [0]
    _t_start = _time.time()
    _print_every = max(1, int(0.1 / dt))   # print roughly every 0.1 time units
    project_f2_interval = params.get('project_f2_interval', 0)
    _suppress_rhs_print = [project_f2_interval > 0]  # outer loop handles printing

    _orig_rhs = compute_derivatives_routeC
    def _rhs_with_progress(t, y, params, ga_obj):
        _call_count[0] += 1
        result = _orig_rhs(t, y, params, ga_obj)
        if not _suppress_rhs_print[0]:
            n = _call_count[0]
            if n % (6 * _print_every) == 1:
                elapsed = _time.time() - _t_start
                frac = min(t / t_max, 1.0)
                eta = (elapsed / frac - elapsed) if frac > 0.01 else float('nan')
                print(f"  t={t:.3f}/{t_max}  ({100*frac:.0f}%)  "
                      f"nfev={n}  elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s",
                      flush=True)
        return result

    if project_f2_interval > 0:
        # ---- Sequential integration with F2 correction (DAE-projection) ----
        # Two modes controlled by project_f2_mode:
        #
        # 'algebraic': solve F2(Λ*)=0 exactly, snap n(ω) → f(ωRR†+Λ*).
        #   Restores F2=0 but BREAKS energy conservation (changes B_mat).
        #
        # 'uniform' (DAE): add δn(ω) = F2 uniformly to all ω.
        #   For symmetric DOS (∫ρω dω = 0), this is the energy-conserving
        #   constraint projection: F2→0, B_mat unchanged, E_phys preserved.
        #   Equivalent to the index-1 DAE stabilization step.
        f2_mode = params.get('project_f2_mode', 'uniform')
        print(f"  F2 correction every {project_f2_interval} step(s), mode='{f2_mode}'")
        H_list_loc  = params['H_list']
        dim_Phi_loc = params['dim_Phi']

        Lmbda_proj = Lmbda_eq_npz.copy()
        y_current  = y0.copy()
        y_out      = np.zeros((len(y0), len(t_eval)))
        y_out[:, 0] = y_current

        for idx in range(1, len(t_eval)):
            t_from = float(t_eval[idx - 1])
            t_to   = float(t_eval[idx])

            sol_seg = solve_ivp(
                _rhs_with_progress, [t_from, t_to], y_current,
                args=(params, ga_obj),
                method=params.get('ode_method', 'RK45'),
                rtol=params.get('ode_rtol', 1e-8),
                atol=params.get('ode_atol', 1e-10),
                max_step=params.get('ode_max_step', np.inf),
                t_eval=[t_to], dense_output=False,
            )
            y_current = sol_seg.y[:, -1]

            # Update Λ warm-start from the accepted step's final state.
            # This is safe (unlike updating inside the rhs) because we only update
            # after each mini-step is fully accepted, not during intermediate stages.
            Phi_ws, n_om_ws = unpack_state(y_current, dim_Phi_loc, N_freq, nqspo)
            Phi_ws /= np.linalg.norm(Phi_ws)
            ffd_ws     = compute_ffdagger_sp(Phi_ws, params)
            Delta_n_ws = safe_Delta(ffd_ws)
            B_mat_ws   = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr, n_om_ws))
            denR_ws    = cr.funcMat(Delta_n_ws, cr.denR)
            fdc_ws     = compute_fdaggerc_sp(Phi_ws, params)
            R_ws       = (fdc_ws.T @ denR_ws).T
            R_vec_ws   = R_ws[:, 0]
            Left_ws    = (R_vec_ws @ B_mat_ws).reshape(nqspo, 1)
            D_ws       = denR_ws @ Left_ws
            Lmbda_ws   = params.get('Lmbda_warm', params.get('Lmbda_eq', np.zeros((nqspo, nqspo)))).copy()
            _n_iter_ws = params.get('n_lambda_iter', 3)
            for _ in range(_n_iter_ws):
                Lc_ws_iter = calc_Lmbdac_fast(Delta_n_ws, D_ws, R_ws,
                                              params['H_list'], params['H_list_arr']) - Lmbda_ws
                H_ws_iter  = build_H_emb_fast(np.real(D_ws[:, 0]), np.real(Lc_ws_iter),
                                              params['H_const'], params['M_D'], params['M_Lc_full'])
                Lmbda_ws   = solve_lambda_differential(
                    Phi_ws, H_ws_iter @ Phi_ws, B_mat_ws, R_vec_ws, ffd_ws, params)
            params['Lmbda_warm'] = Lmbda_ws  # carry forward to next mini-step

            if idx % project_f2_interval == 0:
                Phi_c, n_om_c = unpack_state(y_current, dim_Phi_loc, N_freq, nqspo)
                Phi_c /= np.linalg.norm(Phi_c)
                n_el_c = compute_ffdagger_sp(Phi_c, params)

                if f2_mode == 'uniform':
                    # DAE uniform correction: δn(ω) = F2 for all ω.
                    # ∫ρω δn dω = F2·∫ρω dω = 0 (symmetric DOS) → B_mat unchanged.
                    Delta_n_bath = np.einsum('f,fab->ab', w_rho, np.real(n_om_c))
                    F2_mat = n_el_c - Delta_n_bath          # (nqspo, nqspo) drift
                    n_corr = n_om_c + F2_mat[None, :, :]   # broadcast over ω
                    y_current = pack_state(Phi_c, n_corr)
                else:
                    # Algebraic mode: find Λ* with F2(Λ*)=0, snap n(ω)=f(ωRR†+Λ*)
                    Delta_c = safe_Delta(n_el_c)
                    denR_c  = cr.funcMat(Delta_c, cr.denR)
                    fdc_c   = compute_fdaggerc_sp(Phi_c, params)
                    R_c     = (fdc_c.T @ denR_c).T
                    Lmbda_proj = solve_lambda_from_F2(
                        n_el_c, R_c, Lmbda_proj,
                        omega_arr, w_rho, ga_obj.T, H_list_loc,
                        tol=1e-8, maxiter=50,
                    )
                    RRt_c = R_c @ R_c.T
                    n_proj = np.array([cr.calc_C(RRt_c * om + Lmbda_proj, T=ga_obj.T)
                                       for om in omega_arr])
                    y_current = pack_state(Phi_c, n_proj)

            y_out[:, idx] = y_current

            if idx % _print_every == 0:
                elapsed = _time.time() - _t_start
                frac    = t_to / t_max
                eta     = (elapsed / frac - elapsed) if frac > 0.01 else float('nan')
                print(f"  t={t_to:.3f}/{t_max}  ({100*frac:.0f}%)  "
                      f"nfev={_call_count[0]}  elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s",
                      flush=True)

        print(f"  Integration done. {_call_count[0]} function evaluations.")

        class _MockSol:
            t    = t_eval
            y    = y_out
            nfev = _call_count[0]
        sol = _MockSol()

    else:
        # ---- Standard single solve_ivp ----
        sol = solve_ivp(
            _rhs_with_progress,
            [0.0, t_max],
            y0,
            t_eval=t_eval,
            args=(params, ga_obj),
            method=params.get('ode_method', 'RK45'),
            rtol=params.get('ode_rtol', 1e-8),
            atol=params.get('ode_atol', 1e-10),
            max_step=params.get('ode_max_step', np.inf),
            dense_output=False,
        )
        print(f"  Integration done. {sol.nfev} function evaluations.")

    # ---- Post-processing ----
    print("  Post-processing...")
    dim_Phi     = params['dim_Phi']
    N_freq_p    = params['N_freq']
    op_docc     = params['op_docc']
    op_bb       = params['op_bb']
    op_n_orb_list = params['op_n_orb_list']
    weights_p   = params['weights']
    rho_arr_p   = params['rho_arr']
    omega_arr_p = params['omega_arr']
    w_rho_p     = weights_p * rho_arr_p
    H_list      = params['H_list']
    H_list_arr  = params['H_list_arr']

    docc_list, E_emb_list, E_qp_list, E_lmbda_list, E_phys_list = [], [], [], [], []
    F2_list, TrDelta_list, Z_list, eig_Lc_list, sqrtDtD_list = [], [], [], [], []
    n_orb_lists = [[] for _ in range(len(op_n_orb_list))]

    for k in range(sol.y.shape[1]):
        Phi, n_om = unpack_state(sol.y[:, k], dim_Phi, N_freq_p, nqspo)
        Phi /= np.linalg.norm(Phi)
        Phi_c = np.conj(Phi)

        # Post-processing: differential Route C Λ (same deterministic logic as ODE)
        ffd       = compute_ffdagger_sp(Phi, params)   # n_el from Φ
        Delta_n_p = safe_Delta(ffd)
        B_mat_p   = np.real(np.einsum('f,f,fab->ab', w_rho_p, omega_arr_p, n_om))
        denR_np   = cr.funcMat(Delta_n_p, cr.denR)
        fdc_p     = compute_fdaggerc_sp(Phi, params)
        R_p       = (fdc_p.T @ denR_np).T
        R_vec_p   = R_p[:, 0]

        Left_p    = (R_vec_p @ B_mat_p).reshape(nqspo, 1)
        D_p       = denR_np @ Left_p

        # Solve Λ via differential constraint (Λ=0 seed, diagonal-zero gauge)
        Lc_trial  = calc_Lmbdac_fast(Delta_n_p, D_p, R_p, H_list, H_list_arr)
        H_emb_trial_p = build_H_emb_fast(np.real(D_p[:, 0]), np.real(Lc_trial),
                                          params['H_const'], params['M_D'], params['M_Lc_full'])
        lmbda_pp  = solve_lambda_differential(Phi, H_emb_trial_p @ Phi, B_mat_p,
                                               R_vec_p, ffd, params)

        Lc_p      = calc_Lmbdac_fast(Delta_n_p, D_p, R_p, H_list, H_list_arr) - lmbda_pp
        H_emb_p   = build_H_emb_fast(np.real(D_p[:, 0]), np.real(Lc_p),
                                      params['H_const'], params['M_D'], params['M_Lc_full'])

        E_emb_v  = np.real(Phi_c @ (H_emb_p @ Phi))
        docc_v   = np.real(Phi_c @ (op_docc @ Phi))
        E_qp_v   = 2.0 * np.real(np.trace(R_p @ R_p.T @ B_mat_p))
        E_lmbda_v = np.real(np.trace(lmbda_pp @ Delta_n_p))
        E_phys_v = E_qp_v + U_final * docc_v

        # F2 = ||n_om integral - n_el_phi|| (drift metric)
        Delta_nom_p = np.einsum('f,fab->ab', w_rho_p, np.real(n_om))
        F2_v = np.linalg.norm(ffd - Delta_nom_p)
        TrDelta_v = np.real(np.trace(Delta_n_p))

        Z_v = ga_obj.calc_Z(Lc_p, R_p)

        docc_list.append(docc_v)
        E_emb_list.append(E_emb_v)
        E_qp_list.append(E_qp_v)
        E_lmbda_list.append(E_lmbda_v)
        E_phys_list.append(E_phys_v)
        F2_list.append(F2_v)
        TrDelta_list.append(TrDelta_v)
        Z_list.append(Z_v)
        eig_Lc_list.append(np.linalg.eigvalsh(Lc_p))
        sqrtDtD_list.append(np.sqrt(np.maximum(0.0, np.linalg.eigvalsh(D_p @ D_p.T))))

        for idx, op in enumerate(op_n_orb_list):
            n_orb_lists[idx].append(np.real(Phi_c @ (op @ Phi)))

    return {
        't':       sol.t,
        'docc':    np.array(docc_list),
        'E_emb':   np.array(E_emb_list),
        'E_qp':    np.array(E_qp_list),
        'E_lmbda': np.array(E_lmbda_list),
        'E_phys':  np.array(E_phys_list),
        'F2':      np.array(F2_list),
        'TrDelta': np.array(TrDelta_list),
        'Z':       np.array(Z_list),
        'n_orb':   np.array(n_orb_lists),
        'eig_Lc':  np.array(eig_Lc_list),
        'sqrtDtD': np.array(sqrtDtD_list),
        'E_tot':   np.array(E_emb_list),
    }


# ============================================================
# Route C (Direct F2): ODE over Φ only, n(ω) reconstructed
# ============================================================
def compute_derivatives_routeC_directF2(t, y, params, _lambda_state):
    """
    Route C with direct F2=0 solve at each ODE step.

    ODE state: Φ only (packed as [Re(Φ), Im(Φ)]).
    n(ω) is NOT an ODE state variable; it is reconstructed at each step as
      n(ω) = f(ω RR† + Λ)
    where Λ is solved from ∫ρ f(ω RR† + Λ)dω = ⟨f†f⟩_Φ (F2=0).

    _lambda_state: mutable list [Lmbda_prev] for warm-starting the Λ solve.
    """
    dim_Phi    = params['dim_Phi']
    nqspo      = params['nqspo']
    omega_arr  = params['omega_arr']
    w_rho      = params['weights'] * params['rho_arr']
    M_D        = params['M_D']
    M_Lc_full  = params['M_Lc_full']
    H_const    = params['H_const']
    H_list     = params['H_list']
    H_list_arr = params['H_list_arr']
    T_bath     = params['T_bath']
    N_freq     = params['N_freq']

    Phi = y[:dim_Phi] + 1j * y[dim_Phi:]
    Phi /= np.linalg.norm(Phi)

    # Step 1: n_el = ⟨f†f⟩_Φ
    ffd     = compute_ffdagger_sp(Phi, params)
    Delta_n = safe_Delta(ffd)

    # Step 2: R
    denR_mat = cr.funcMat(Delta_n, cr.denR)
    fdc      = compute_fdaggerc_sp(Phi, params)
    R        = (fdc.T @ denR_mat).T
    R_vec    = R[:, 0]

    # Step 3: Solve F2=0 directly (warm-started from previous Λ)
    tol_lam = params.get('lambda_tol', 1e-8)
    Lmbda = solve_lambda_from_F2(
        ffd, R, _lambda_state[0],
        omega_arr, w_rho, T_bath, H_list,
        tol=tol_lam, maxiter=params.get('lambda_maxiter', 100),
    )
    _lambda_state[0] = Lmbda

    # Step 4: Reconstruct n(ω) = f(ωRR† + Λ)
    RRt     = R_vec[:, None] * R_vec[None, :]
    n_omega = np.array([cr.calc_C(omega_arr[f] * RRt + Lmbda, T=T_bath)
                        for f in range(N_freq)])

    # Step 5: B_mat, D
    B_mat = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr, n_omega))
    Left  = (R_vec @ B_mat).reshape(nqspo, 1)
    D     = denR_mat @ Left

    # Step 6: H_emb
    Lmbdac = calc_Lmbdac_fast(Delta_n, D, R, H_list, H_list_arr) - Lmbda
    H_emb  = build_H_emb_fast(np.real(D[:, 0]), np.real(Lmbdac), H_const, M_D, M_Lc_full)

    # Step 7: dΦ/dt = -i H_emb Φ
    dPhi_dt = -1j * (H_emb @ Phi)
    return np.concatenate([np.real(dPhi_dt), np.imag(dPhi_dt)])


def run_full_simulation_routeC_directF2(ga_obj, U_final, t_max, dt,
                                        N_freq=100, extra_params=None):
    """
    Route C (Direct F2) quench simulation.

    Same initial state as Route C (full gGA equilibrium).
    ODE state is Φ only; n(ω) maintained exactly on F2=0 manifold via
    direct least-squares solve at each RHS evaluation.
    """
    import time as _time
    from scipy.integrate import solve_ivp

    params = prepare_full_params(ga_obj, U_final, N_freq)
    if extra_params:
        params.update(extra_params)
    params['T_bath'] = ga_obj.T

    nqspo     = ga_obj.nqspo
    omega_arr = params['omega_arr']
    weights   = params['weights']
    rho_arr   = params['rho_arr']
    w_rho     = weights * rho_arr
    T_bath    = ga_obj.T

    # ---- Same initialization as Route C ----
    Phi_0  = params['Phi_0'].copy()
    n_el_0 = compute_ffdagger_sp(Phi_0, params)
    fdc_0  = compute_fdaggerc_sp(Phi_0, params)
    R0_npz = (fdc_0.T @ cr.funcMat(safe_Delta(n_el_0), cr.denR)).T

    print(f"  初期 |Φ_0⟩ ノルム: {np.linalg.norm(Phi_0):.6f}")
    print(f"  R0 (npz): {np.round(R0_npz[:, 0], 4)}")

    eig_n, U_npz = np.linalg.eigh(n_el_0)
    eig_Lmbda    = np.linalg.eigvalsh(np.real(ga_obj.Lmbda))
    Lmbda_guess  = U_npz @ np.diag(eig_Lmbda[::-1]) @ U_npz.T
    Lmbda_eq_npz = solve_lambda_from_F2(
        n_el_0, R0_npz, Lmbda_guess,
        omega_arr, w_rho, T_bath, params['H_list'],
        tol=1e-10, maxiter=500,
    )
    params['Lmbda_eq'] = Lmbda_eq_npz
    print(f"  Λ_eq 固有値 (npz basis): {np.round(np.linalg.eigvalsh(Lmbda_eq_npz), 4)}")

    RRt0     = R0_npz @ R0_npz.T
    n0_omega = np.array([cr.calc_C(RRt0 * om + Lmbda_eq_npz, T=T_bath)
                         for om in omega_arr])
    Delta0   = np.einsum('f,fab->ab', w_rho, np.real(n0_omega))
    F2_init  = np.linalg.norm(n_el_0 - Delta0)
    print(f"  初期 F2 = {F2_init:.3e}")

    # ---- ODE: Φ only ----
    y0 = np.concatenate([np.real(Phi_0), np.imag(Phi_0)])
    t_eval = np.arange(0.0, t_max, dt)

    _lambda_state = [Lmbda_eq_npz.copy()]

    print(f"  State size: {len(y0)} reals (Φ only, no n(ω))")
    print(f"  Integrating from t=0 to t={t_max} (Route C Direct F2)...")

    _call_count = [0]
    _t_start    = _time.time()
    _print_every = max(1, int(0.1 / dt))

    def _rhs_with_progress(t, y):
        _call_count[0] += 1
        result = compute_derivatives_routeC_directF2(t, y, params, _lambda_state)
        n = _call_count[0]
        if n % (6 * _print_every) == 1:
            elapsed = _time.time() - _t_start
            frac = min(t / t_max, 1.0)
            eta = (elapsed / frac - elapsed) if frac > 0.01 else float('nan')
            print(f"  t={t:.3f}/{t_max}  ({100*frac:.0f}%)  "
                  f"nfev={n}  elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s", flush=True)
        return result

    sol = solve_ivp(
        _rhs_with_progress,
        [0.0, t_max],
        y0,
        t_eval=t_eval,
        method=params.get('ode_method', 'RK45'),
        rtol=params.get('ode_rtol', 1e-8),
        atol=params.get('ode_atol', 1e-10),
        dense_output=False,
    )
    print(f"  Integration done. {sol.nfev} function evaluations.")

    # ---- Post-processing ----
    print("  Post-processing...")
    dim_Phi       = params['dim_Phi']
    op_docc       = params['op_docc']
    op_n_orb_list = params['op_n_orb_list']
    H_list        = params['H_list']
    H_list_arr    = params['H_list_arr']
    N_freq_p      = N_freq

    docc_list, E_emb_list, E_qp_list, E_lmbda_list, E_phys_list = [], [], [], [], []
    F2_list, Z_list = [], []
    n_orb_lists = [[] for _ in range(len(op_n_orb_list))]

    Lmbda_pp = Lmbda_eq_npz.copy()
    for k in range(sol.y.shape[1]):
        Phi   = sol.y[:dim_Phi, k] + 1j * sol.y[dim_Phi:, k]
        Phi  /= np.linalg.norm(Phi)
        Phi_c = np.conj(Phi)

        ffd_k   = compute_ffdagger_sp(Phi, params)
        Delta_k = safe_Delta(ffd_k)
        fdc_k   = compute_fdaggerc_sp(Phi, params)
        denR_k  = cr.funcMat(Delta_k, cr.denR)
        R_k     = (fdc_k.T @ denR_k).T
        R_vec_k = R_k[:, 0]

        Lmbda_pp = solve_lambda_from_F2(
            ffd_k, R_k, Lmbda_pp,
            omega_arr, w_rho, T_bath, H_list,
            tol=1e-10, maxiter=200,
        )

        RRt_k   = R_vec_k[:, None] * R_vec_k[None, :]
        n_om_k  = np.array([cr.calc_C(omega_arr[f] * RRt_k + Lmbda_pp, T=T_bath)
                             for f in range(N_freq_p)])
        B_k     = np.real(np.einsum('f,f,fab->ab', w_rho, omega_arr, n_om_k))
        Left_k  = (R_vec_k @ B_k).reshape(nqspo, 1)
        D_k     = denR_k @ Left_k

        Lmbdac_k = calc_Lmbdac_fast(Delta_k, D_k, R_k, H_list, H_list_arr) - Lmbda_pp
        H_emb_k  = build_H_emb_fast(np.real(D_k[:, 0]), np.real(Lmbdac_k),
                                     params['H_const'], params['M_D'], params['M_Lc_full'])

        docc_k   = np.real(Phi_c @ (op_docc @ Phi))
        E_emb_k  = np.real(Phi_c @ (H_emb_k @ Phi))
        E_qp_k   = 2.0 * np.real(np.trace(R_k @ R_k.T @ B_k))
        E_lam_k  = np.real(np.trace(Lmbda_pp @ Delta_k))
        E_phys_k = E_qp_k + U_final * docc_k

        Delta_bath_k = np.einsum('f,fab->ab', w_rho, np.real(n_om_k))
        F2_k = np.linalg.norm(ffd_k - Delta_bath_k)

        sqrtDtD_k = cr.funcMat(Delta_k, cr.denR)
        Z_k = 2.0 * np.real(np.einsum('a,ab,b', np.conj(R_vec_k), sqrtDtD_k, R_vec_k))

        docc_list.append(docc_k)
        E_emb_list.append(E_emb_k)
        E_qp_list.append(E_qp_k)
        E_lmbda_list.append(E_lam_k)
        E_phys_list.append(E_phys_k)
        F2_list.append(F2_k)
        Z_list.append(Z_k)
        for i, op in enumerate(op_n_orb_list):
            n_orb_lists[i].append(np.real(Phi_c @ (op @ Phi)))

    res = dict(
        t        = sol.t,
        docc     = np.array(docc_list),
        E_emb    = np.array(E_emb_list),
        E_qp     = np.array(E_qp_list),
        E_lmbda  = np.array(E_lmbda_list),
        E_phys   = np.array(E_phys_list),
        F2       = np.array(F2_list),
        Z        = np.array(Z_list),
        n_orb    = np.array(n_orb_lists),
    )

    E     = res['E_phys']
    F2    = res['F2']
    rel_E = np.abs((E - E[0]) / max(abs(E[0]), 1e-12))
    print(f"  F2_max={F2.max():.3e}")
    print(f"  |ΔE/E|_max={rel_E.max():.3e}")
    return res

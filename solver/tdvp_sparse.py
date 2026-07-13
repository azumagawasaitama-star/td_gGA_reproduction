"""
tdvp_sparse.py

td_gGA_solver.py が使う基礎ヘルパー関数群:
  - compute_ffdagger_sp: |Φ⟩から⟨f†f⟩（電子密度行列）を計算
  - solve_lambda_from_F2: 拘束 ∫ρf(ωRR†+Λ)=ffd_target を満たすΛを求める
    （t=0の初期条件Λ_eq計算、および静的解との照合に使用）

これらの関数はもともと Route C（このファイル自身の独自TD時間発展ドライバ、
Λ(t)を毎ステップ動的に解く方式）のために書かれた。Route C のドライバ本体は
B=3以上でエネルギー非保存という既知の問題があり使われておらず、正しい実装は
td_gGA_solver.py にある。ドライバ本体は削除済み
（検証時の経緯は POSTMORTEM_2026-07-07.md, derivation_F2_conservation_proof.md 参照）。
"""
import numpy as np
from scipy.optimize import least_squares
import convenience_routines as cr


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

"""
td_gGA_solver.py  (main TD-gGA solver; 2026-07-07 に導出・検証)
========================================
論文 (Guerci-Capone-Lanata PRR 5, L032023) の転置規約に完全準拠した複素 TDVP ソルバー。
導出と F2 厳密保存の証明: derivation_F2_conservation_proof.md
既存ソルバーには一切触れない（新規ファイルで完結）。

コードの n(ω) は標準 Fermi 行列 n = f(ωRR†+Λ), n_ab = ⟨η†_b η_a⟩（論文の n の転置）。
この規約に翻訳した一貫式系:

  Δh    = ⟨f f†⟩  (hole 行列, Δh_ij = ⟨f_j f†_i⟩ = raw op_bb 期待値)   [= 論文の Δ]
  g     = [Δh(1−Δh)]^{1/2}
  R     = conj(g⁻¹ ⟨f†c⟩)
  B     = Σ_f w_f ω_f n_f   (複素エルミート, real() 禁止)
  D     = g⁻¹ conj(B R)                                  [論文 Eq.18, c†f 側の係数]
  M     = D Rᵀ + R* D†      (エルミート)
  Λc    = −(dg[M])ᵀ         (Loewner 閉形式; 完全エルミート基底射影と等価)
  H_emb = H_const + Σ_p [conj(D_p) f†_p c + D_p c† f_p] + Σ_pq Λc_pq f_q f†_p
  ∂t Φ  = −i H_emb Φ
  ∂t n  = −iω [RR†, n]      (符号注意: 従来 'vn' 形は逆符号)
  F2    = ‖Δh − (Σ_f w_f n_f)ᵀ‖   (転置必須) → 厳密保存（証明済み）

初期条件: 全 gGA 平衡 Φ0 から
  Λ_eq: Σ_f w_f f(ω_f R0R0† + Λ) = Δh(Φ0) = I − ffd(Φ0) を解く（従来は = ffd で符号逆）
  n0   = f(ω R0R0† + Λ_eq)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import time as _time
import convenience_routines as cr
from tdvp_core import prepare_full_params, pack_state, unpack_state, compute_fdaggerc_sp
from tdvp_sparse import compute_ffdagger_sp, solve_lambda_from_F2


# ---------- 期待値ヘルパ（複素のまま; dense/sparse 両対応） ----------
def hole_cplx(Phi, params):
    """Δh_ij = ⟨f_j f†_i⟩ = raw op_bb 期待値（スピン平均, 複素エルミート）"""
    nq = params['nqspo']
    Pc = np.conj(Phi)
    if params.get('sparse'):
        M = params['M_bb']   # M_bb[i][j] = op_bb[2i,2j]+op_bb[2i+1,2j+1]（スピン和済み疎行列）
        return np.array([[0.5 * (Pc @ (M[i][j] @ Phi)) for j in range(nq)]
                         for i in range(nq)])
    op_bb = params['op_bb']
    return np.array([[0.5*(Pc @ (op_bb[2*i, 2*j] @ Phi) + Pc @ (op_bb[2*i+1, 2*j+1] @ Phi))
                      for j in range(nq)] for i in range(nq)])


def fdc_cplx(Phi, params):
    """⟨f†c⟩ 複素のまま (nq,1)"""
    nq = params['nqspo']
    Pc = np.conj(Phi)
    if params.get('sparse'):
        return np.array([[Pc @ (params['M_bp'][i] @ Phi)] for i in range(nq)]) * 0.5
    op_bp = params['op_bp']
    return np.array([[Pc @ (op_bp[2*i, 0] @ Phi) + Pc @ (op_bp[2*i+1, 1] @ Phi)]
                     for i in range(nq)]) * 0.5


# ---------- 行列関数（複素エルミート安全, eigh ベース） ----------
_CLIP = 1e-12

def _eigh_clip(Delta):
    Dh = 0.5 * (Delta + Delta.conj().T)
    ev, V = np.linalg.eigh(Dh)
    ev = np.clip(ev.real, _CLIP, 1.0 - _CLIP)
    return ev, V


def g_inv_of(Delta):
    """[Δ(1−Δ)]^{-1/2}"""
    ev, V = _eigh_clip(Delta)
    return (V * (1.0 / np.sqrt(ev * (1.0 - ev)))) @ V.conj().T


def loewner_dg(Delta, M):
    """dg[M]: g(Δ)=√(Δ(1−Δ)) の Δ における Fréchet 微分（方向 M, 複素エルミート）"""
    ev, V = _eigh_clip(Delta)
    g = np.sqrt(ev * (1.0 - ev))
    gp = (0.5 - ev) / g                       # g'(x)
    dl = ev[:, None] - ev[None, :]
    K = np.where(np.abs(dl) > 1e-9, (g[:, None] - g[None, :]) / np.where(np.abs(dl) > 1e-9, dl, 1.0),
                 0.5 * (gp[:, None] + gp[None, :]))
    Mb = V.conj().T @ M @ V
    return V @ (K * Mb) @ V.conj().T


# ---------- 一貫規約での R, D, Λc, H_emb ----------
def compute_RDLc_paper(Phi, n_omega, params):
    """電子規約（このコードベースの静的鞍点の実際の規約）で一貫した R, D, Λc。

    診断 diag_conv.py で確定した事実: 静的解は ∫ρf(ωRRᵀ+Λ) = ⟨f†f⟩（電子）を満たす。
    電子規約での F2 厳密保存条件（導出は derivation_F2_conservation_proof.md 追記参照）:
      R  = g⁻¹ fdc,  g = [Δe(1−Δe)]^{1/2},  Δe = ⟨f†f⟩
      D  = conj(g⁻¹ B R)          （H_emb の f†c スロット係数）
      Λc = +(dg_{Δe}[M])ᵀ,  M = R Dᵀ + D* R†
    """
    nq = params['nqspo']; omega_arr = params['omega_arr']
    w_rho = params['weights'] * params['rho_arr']
    ffd = np.eye(nq) - hole_cplx(Phi, params)            # Δe = ⟨f†f⟩（電子, 複素エルミート）
    gi = g_inv_of(ffd)
    R = gi @ fdc_cplx(Phi, params)                       # 共役なし
    Rv = R[:, 0]
    B_mat = np.einsum('f,f,fab->ab', w_rho, omega_arr, n_omega)   # 複素のまま
    D = np.conj(gi @ (B_mat @ R))
    K = R @ D.T
    M = K + K.conj().T                                   # R Dᵀ + D* R†
    Lc = (loewner_dg(ffd, M)).T                          # Λc = +(dg_{Δe}[M])ᵀ
    return R, Rv, D, Lc, B_mat, ffd


def build_H_emb_paper(D_vec, Lmbdac, params):
    """H = H_const + Σ_p [D_p f†_p c + conj(D_p) c† f_p] + Σ_pq Λc_pq f_q f†_p"""
    op_bp = params['op_bp']; nq = params['nqspo']
    H = params['H_const'].astype(complex).copy()
    for p in range(nq):
        tmp = op_bp[2*p, 0] + op_bp[2*p+1, 1]            # f†_p c（実行列）
        H += D_vec[p] * tmp + np.conj(D_vec[p]) * tmp.T  # D は f†c スロットの係数
    H += np.einsum('pq,pqij->ij', Lmbdac, params['M_Lc_full'])   # Σ Λc_pq f_q f†_p
    return H


def apply_H_emb_sp(D_vec, Lmbdac, Phi, params):
    """疎行列パス: H を組み立てずに H·Φ を疎行列×ベクトル積の和で直接評価
    （dense の build_H_emb_paper と厳密に同一の演算子）"""
    nq = params['nqspo']
    out = params['H_const'] @ Phi
    for p in range(nq):
        out = out + D_vec[p] * (params['M_bp'][p] @ Phi) \
                  + np.conj(D_vec[p]) * (params['M_bp_T'][p] @ Phi)
        for q in range(nq):
            c = Lmbdac[p, q]
            if abs(c) > 1e-15:
                out = out + c * (params['M_bb'][p][q] @ Phi)
    return out


def prepare_sparse_params(ga, U_final, N_freq):
    """B≥7 用: 演算子を疎行列のまま保持する軽量 params（dense 版と同じキー体系の必要分のみ）"""
    import scipy.sparse as sp
    from math import comb as _comb
    from tdvp_core import setup_frequency_grid
    ed = ga.imp_solver
    dim_Phi = ed.hsize_half; nq = ga.nqspo
    imp_nr = ed.impurity_nr; imp_type = ed.impurity_type
    ioff = sum(int(_comb(ed.n_tot_orb, i)) for i in range(ed.n_half))
    iend = ioff + dim_Phi

    def _load(pref, i, j):
        return sp.load_npz(f"{pref}_imp-{imp_nr}_{imp_type}_op+{i}-{j}.npz").tocsr()

    print(f"  [sparse] loading operators (dim_Phi={dim_Phi}, nq={nq})...", flush=True)
    M_bp = [(_load('bath-phys', 2*p, 0) + _load('bath-phys', 2*p+1, 1)).tocsr()
            for p in range(nq)]
    M_bp_T = [m.T.tocsr() for m in M_bp]
    M_bb = [[(_load('bath-bath', 2*p, 2*q) + _load('bath-bath', 2*p+1, 2*q+1)).tocsr()
             for q in range(nq)] for p in range(nq)]
    FH = ed.build_creation_ops()
    n_up = FH[0] @ FH[0].conj().T
    n_dn = FH[1] @ FH[1].conj().T
    op_docc = (n_up @ n_dn)[ioff:iend, ioff:iend].tocsr()
    D0 = np.zeros((nq, 1)); Lc0 = np.zeros((nq, nq))
    H_const = ed.build_Hemb(D0, np.array([[-U_final/2.0]]), Lc0, U_final).tocsr()
    omega_arr, weights, rho_arr = setup_frequency_grid(N_freq)
    print(f"  [sparse] done. H_const nnz={H_const.nnz}", flush=True)
    return dict(sparse=True, dim_Phi=dim_Phi, nqspo=nq, N_freq=N_freq,
                M_bp=M_bp, M_bp_T=M_bp_T, M_bb=M_bb, op_docc=op_docc,
                H_const=H_const, Phi_0=np.asarray(ed.eig_vec).copy(),
                omega_arr=omega_arr, weights=weights, rho_arr=rho_arr,
                H_list=ga.H_list)


# ---------- ODE 右辺 ----------
def rhs_paper(t, y, params):
    dim_Phi = params['dim_Phi']; N_freq = params['N_freq']; nq = params['nqspo']
    omega_arr = params['omega_arr']
    Phi, n_omega = unpack_state(y, dim_Phi, N_freq, nq)
    Phi = Phi / np.linalg.norm(Phi)
    R, Rv, D, Lc, B_mat, Dh = compute_RDLc_paper(Phi, n_omega, params)
    if params.get('sparse'):
        dPhi = -1j * apply_H_emb_sp(D[:, 0], Lc, Phi, params)
    else:
        H = build_H_emb_paper(D[:, 0], Lc, params)
        dPhi = -1j * (H @ Phi)
    RRt = np.outer(Rv, np.conj(Rv))                      # (RR†)_ab = R_a R_b*
    dn = -1j * omega_arr[:, None, None] * (RRt[None] @ n_omega - n_omega @ RRt[None])
    return pack_state(dPhi, dn)


# ---------- 静的計算とキャッシュ ----------
# static_cache/, data/ はプロジェクトルート直下で共有（td_gGA_base と studies/ の両方から使う）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_CACHE_DIR = os.path.join(_ROOT, 'static_cache')
DATA_DIR = os.path.join(_ROOT, 'data')


def _static_cache_path(B, U_i, T_gGA):
    return os.path.join(STATIC_CACHE_DIR, f"B{B}_Ui{U_i:g}_T{T_gGA:g}.npz")


def save_static(ga, B, U_i, T_gGA):
    """収束済み静的解の必要成分をディスクへ（数MB以下）"""
    os.makedirs(STATIC_CACHE_DIR, exist_ok=True)
    ed = ga.imp_solver
    np.savez(_static_cache_path(B, U_i, T_gGA),
             eig_vec=np.asarray(ed.eig_vec), fdaggerc=np.asarray(ed.fdaggerc),
             ffdagger=np.asarray(ed.ffdagger), Lmbda=np.asarray(ga.Lmbda),
             D=np.asarray(ga.D), Lmbdac=np.asarray(ga.Lmbdac), R=np.asarray(ga.R),
             B=B, U_i=U_i, T_gGA=T_gGA)


def load_static(B, U_i=0.05, T_gGA=0.003):
    """キャッシュから静的解を復元（GA を最適化なしで作り、収束値を注入）。
    GA 生成時に演算子 npz が正しい B で再生成される点も担保される。"""
    path = _static_cache_path(B, U_i, T_gGA)
    if not os.path.exists(path):
        return None
    from gga_static_solver import GA
    ga = GA(U=U_i, nghost=2 * (B - 1), nphysorb=2, n=0.5, T=T_gGA, eks=-99)
    z = np.load(path)
    ga.Lmbda, ga.D, ga.Lmbdac, ga.R = z['Lmbda'], z['D'], z['Lmbdac'], z['R']
    ed = ga.imp_solver
    ed.eig_vec, ed.fdaggerc, ed.ffdagger = z['eig_vec'], z['fdaggerc'], z['ffdagger']
    return ga


def get_static(B, U_i=0.05, T_gGA=0.003, verbose=True):
    """キャッシュがあれば load、なければ solve して save"""
    ga = load_static(B, U_i, T_gGA)
    if ga is not None:
        if verbose:
            print(f"  static cache hit: B={B} U_i={U_i} T={T_gGA}", flush=True)
        return ga
    ga = solve_static(B, U_i, T_gGA)
    save_static(ga, B, U_i, T_gGA)
    return ga


def save_run(rec, B, U_i, U_f, subdir=None):
    """TD 実行結果を data/B{B}/ に1ファイルで保存"""
    d = os.path.join(DATA_DIR, subdir or f"B{B}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"quench_Ui{U_i:g}_Uf{U_f:g}.npz")
    np.savez(path, **{k: np.array(v) for k, v in rec.items()},
             B=B, U_i=U_i, U_f=U_f)
    return path


# ---------- 実行 ----------
def solve_static(B, U_i=0.05, T_gGA=0.003):
    """静的 gGA 平衡解（δU 間で使い回すため分離）"""
    from gga_static_solver import GA
    nq = B
    ga = GA(U=U_i, nghost=2 * (B - 1), nphysorb=2, n=0.5, T=T_gGA, eks=-99)
    nhl = nq * (nq + 1) // 2
    linit = np.zeros(nhl); linit[0] = 1.0
    rinit = np.zeros(nq); rinit[0] = 1.0
    ga.optimize_selfc(rinit=rinit, lambdainit=linit, muinit=0.0)
    return ga


def run(B, U_i=0.05, U_f=2.5, T_gGA=0.003, N_freq=60, t_max=10.0, dt=0.01,
        verbose=True, tag="", ga=None, ic_mod=None, sparse=None,
        adaptive=True, rtol=1e-8, atol=1e-10, method='DOP853'):
    """adaptive=True（既定）: scipy solve_ivp の適応刻み積分器を使用
    （論文が使った RKSUITE と同じく適応刻み。固定 dt=0.01 の RK4 は
    クエンチ直後の鋭い過渡（B=7 で確認済み）を追いきれず軌道が破綻することがある）。
    adaptive=False: 旧来の固定刻み RK4（dt 引数を使用、比較用に残す）。
    dt は adaptive=True では出力サンプリング間隔としてのみ使う（積分刻みではない）。"""
    nq = B
    if ga is None:
        ga = solve_static(B, U_i, T_gGA)

    if sparse is None:
        sparse = (B >= 6)     # B=7 以上は密行列が ~1.3GB/個 になるため疎行列パス
    if sparse:
        params = prepare_sparse_params(ga, U_f, N_freq)
    else:
        params = prepare_full_params(ga, U_f, N_freq)
    omega_arr = params['omega_arr']; w_rho = params['weights'] * params['rho_arr']
    dim_Phi = params['dim_Phi']; H_list = params['H_list']

    Phi_0 = params['Phi_0'].copy().astype(complex)
    Phi_0 /= np.linalg.norm(Phi_0)

    # --- 初期条件（電子規約: 静的鞍点そのもの） ---
    ffd0 = np.eye(nq) - np.real(hole_cplx(Phi_0, params))   # Φ0 は実 → 実対称
    fdc0 = np.real(fdc_cplx(Phi_0, params))
    gi0 = g_inv_of(ffd0)
    R0 = np.real(gi0 @ fdc0)                             # 実
    RRt0 = R0 @ R0.T
    if B > 1:
        # 真の Λ_eq = 静的解のゲージ回転（root-finding は別の根を拾うため使わない）
        # fix_gauge: imp 系 → fg(npz) 系の変換 W = permmat @ phasemat @ transmat.T
        Leq = None
        try:
            _, _, phasemat, permmat, transmat = ga.fix_gauge(
                np.real(ga.D), np.real(ga.Lmbdac), lfor_D=True, lreturn_mats=True)
            W = permmat @ phasemat @ transmat.T
            R_imp = np.real(ga.R).reshape(nq, 1)
            err_R = np.linalg.norm(W @ R_imp - R0)
            Leq_rot = W @ np.real(ga.Lmbda) @ W.T
            if verbose:
                print(f"  [{tag}] gauge-rot check: ‖W·R_imp − R0‖={err_R:.2e} "
                      f"Λrot_eig={np.round(np.linalg.eigvalsh(Leq_rot), 3)}", flush=True)
            if err_R < 5e-2:
                # ±Λrot の両方で F2_init を測り小さい方を採用。
                # （Λ→−Λ は補集合恒等式 ∫ρf(ωM−Λ)=I−∫ρf(ωM+Λ) で electron/hole の
                #   ペアリング反転を吸収する。B=5 で +Λrot が hole 側に落ちる事例あり）
                w_rho_ic = params['weights'] * params['rho_arr']
                def _f2_of(L):
                    nn = np.einsum('f,fab->ab', w_rho_ic, np.array(
                        [cr.calc_C(omega_arr[f] * RRt0 + L, T=ga.T) for f in range(N_freq)]))
                    return float(np.linalg.norm(ffd0 - nn))
                cands = [(Leq_rot, _f2_of(Leq_rot)), (-Leq_rot, _f2_of(-Leq_rot))]
                (Leq, f2_best) = min(cands, key=lambda x: x[1])
                if verbose:
                    print(f"  [{tag}] Λ_eq 符号選択: F2(+Λrot)={cands[0][1]:.2e} "
                          f"F2(−Λrot)={cands[1][1]:.2e} → {'+' if Leq is cands[0][0] else '−'}",
                          flush=True)
                if f2_best > 1e-3:
                    if verbose:
                        print(f"  [{tag}] F2_init={f2_best:.2e} > 1e-3 → root-refine", flush=True)
                    Leq = solve_lambda_from_F2(ffd0, R0, Leq, omega_arr, w_rho_ic, ga.T,
                                               H_list, tol=1e-10, maxiter=500)
        except Exception as e:
            if verbose:
                print(f"  [{tag}] gauge-rot 失敗 ({e}) → root-finding にフォールバック", flush=True)
        if Leq is None:
            eig_n, U_n = np.linalg.eigh(ffd0)
            eig_L = np.linalg.eigvalsh(np.real(ga.Lmbda))
            Lg = U_n @ np.diag(eig_L[::-1]) @ U_n.T      # 占有小 ↔ Λ大 のペアリング
            Leq = solve_lambda_from_F2(ffd0, R0, Lg, omega_arr, w_rho, ga.T, H_list,
                                       tol=1e-10, maxiter=500)
    else:
        Leq = np.zeros((1, 1))
    if ic_mod is not None:
        # 初期条件の人工変形フック（縮退多様体探索用）: (Phi_0, Leq) を差し替え
        Phi_0, Leq = ic_mod(Phi_0, Leq)
        Phi_0 = Phi_0 / np.linalg.norm(Phi_0)
        ffd0 = np.eye(nq) - np.real(hole_cplx(Phi_0, params))
        fdc0 = np.real(fdc_cplx(Phi_0, params))
        gi0 = g_inv_of(ffd0)
        R0 = np.real(gi0 @ fdc0)
        RRt0 = R0 @ R0.T
    n0 = np.array([cr.calc_C(omega_arr[f] * RRt0 + Leq, T=ga.T) for f in range(N_freq)])
    F2_init = np.linalg.norm(ffd0 - np.einsum('f,fab->ab', w_rho, n0))

    docc0 = float(np.real(np.conj(Phi_0) @ (params['op_docc'] @ Phi_0)))
    RRt0c = np.outer(R0[:, 0], np.conj(R0[:, 0]))
    B0 = np.einsum('f,f,fab->ab', w_rho, omega_arr, n0)
    E0 = 2.0 * float(np.real(np.einsum('ab,ba', RRt0c, B0))) + U_f * docc0
    if verbose:
        print(f"  [{tag}] init: d0={docc0:.4f} E0={E0:.6f} F2_init={F2_init:.2e} "
              f"Λeq_eig={np.round(np.linalg.eigvalsh(Leq), 3)}", flush=True)

    y = pack_state(Phi_0, n0)
    n_steps = int(round(t_max / dt))
    rec = dict(t=[], d=[], E=[], dE=[], F2=[], sqDtD=[], Dsplit=[], eigLc=[])
    _t0 = _time.time()

    def observe(y, t):
        Phi, n_om = unpack_state(y, dim_Phi, N_freq, nq)
        Phi = Phi / np.linalg.norm(Phi)
        R, Rv, D, Lc, Bm, ffd_t = compute_RDLc_paper(Phi, n_om, params)
        RRt = np.outer(Rv, np.conj(Rv))
        docc = float(np.real(np.conj(Phi) @ (params['op_docc'] @ Phi)))
        E = 2.0 * float(np.real(np.einsum('ab,ba', RRt, Bm))) + U_f * docc
        F2 = float(np.linalg.norm(ffd_t - np.einsum('f,fab->ab', w_rho, n_om)))
        rec['t'].append(t); rec['d'].append(docc); rec['E'].append(E)
        rec['dE'].append(abs(E - E0) / max(abs(E0), 1e-12))
        rec['F2'].append(F2)
        rec['sqDtD'].append(float(np.sqrt(np.sum(np.abs(D[:, 0])**2))))
        Dabs = np.sort(np.abs(D[:, 0]))
        rec['Dsplit'].append(float(Dabs[1] - Dabs[0]) if nq > 1 else 0.0)
        rec['eigLc'].append(np.linalg.eigvalsh(0.5 * (Lc + Lc.conj().T)))

    observe(y, 0.0)
    if adaptive:
        from scipy.integrate import solve_ivp
        t_eval = np.linspace(0.0, t_max, n_steps + 1)[1:]

        def _fun(t, yv):
            return rhs_paper(t, yv, params)

        sol = solve_ivp(_fun, (0.0, t_max), y, method=method, t_eval=t_eval,
                        rtol=rtol, atol=atol, dense_output=False)
        if not sol.success:
            print(f"  [{tag}] 警告: solve_ivp 失敗 ({sol.message})", flush=True)
        for i in range(sol.y.shape[1]):
            observe(sol.y[:, i], t_eval[i])
            if verbose and (i + 1) % 200 == 0:
                print(f"    [{tag}] t={t_eval[i]:.1f} d={rec['d'][-1]:.4f} "
                      f"|ΔE/E|={rec['dE'][-1]:.2e} F2={rec['F2'][-1]:.2e} "
                      f"√DtD={rec['sqDtD'][-1]:.3f} nfev={sol.nfev} "
                      f"({_time.time()-_t0:.0f}s)", flush=True)
        if verbose:
            print(f"  [{tag}] solve_ivp 完了: nfev={sol.nfev} njev={sol.njev if hasattr(sol,'njev') else 0} "
                  f"({_time.time()-_t0:.0f}s)", flush=True)
    else:
        for step in range(n_steps):
            tn = step * dt
            k1 = rhs_paper(tn,        y,             params)
            k2 = rhs_paper(tn + dt/2, y + dt/2 * k1, params)
            k3 = rhs_paper(tn + dt/2, y + dt/2 * k2, params)
            k4 = rhs_paper(tn + dt,   y + dt * k3,   params)
            y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            observe(y, (step + 1) * dt)
            if verbose and (step + 1) % 200 == 0:
                print(f"    [{tag}] step {step+1}/{n_steps} t={(step+1)*dt:.1f} "
                      f"d={rec['d'][-1]:.4f} |ΔE/E|={rec['dE'][-1]:.2e} F2={rec['F2'][-1]:.2e} "
                      f"√DtD={rec['sqDtD'][-1]:.3f} ({_time.time()-_t0:.0f}s)", flush=True)
    for k in rec:
        rec[k] = np.array(rec[k])
    rec['E0'] = E0; rec['F2_init'] = F2_init
    return rec


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}

    if mode in ("all", "noquench"):
        print("=== B=3 ノークエンチ (U=0.05→0.05, t=5): 証明の直接テスト ===", flush=True)
        r = run(3, U_f=0.05, t_max=5.0, tag="B3-nq")
        print(f"  結果: d変動={r['d'].max()-r['d'].min():.2e} "
              f"|ΔE/E|max={r['dE'].max():.2e} F2max={r['F2'].max():.2e}")
        out['nq3'] = r

    if mode in ("all", "quench"):
        print("=== B=1 クエンチ回帰 (U=0.05→2.5, t=10) ===", flush=True)
        r1 = run(1, t_max=10.0, tag="B1-q")
        print(f"  結果: |ΔE/E|max={r1['dE'].max():.2e} F2max={r1['F2'].max():.2e} "
              f"d range=[{r1['d'].min():.4f},{r1['d'].max():.4f}]")
        out['q1'] = r1

        print("=== B=3 クエンチ本命 (U=0.05→2.5, t=10) ===", flush=True)
        r3 = run(3, t_max=10.0, tag="B3-q")
        print(f"  結果: |ΔE/E|max={r3['dE'].max():.2e} F2max={r3['F2'].max():.2e} "
              f"d range=[{r3['d'].min():.4f},{r3['d'].max():.4f}] d_mean={r3['d'].mean():.4f}")
        out['q3'] = r3

        print("\n" + "=" * 64)
        print("PAPERCONV SOLVER 判定 (U=0.05→2.5)")
        print("=" * 64)
        print(f"B=1: |ΔE/E|={r1['dE'].max():.2e} F2={r1['F2'].max():.2e}")
        print(f"B=3: |ΔE/E|={r3['dE'].max():.2e} F2={r3['F2'].max():.2e}")
        print(f"     max|d3-d1|={np.abs(r3['d']-r1['d']).max():.3e}  "
              f"d3_mean={r3['d'].mean():.4f} vs d1_mean={r1['d'].mean():.4f}")
        print(f"     √(D†D) B=3: 初期{r3['sqDtD'][0]:.3f} 最小{r3['sqDtD'].min():.3f} "
              f"終値{r3['sqDtD'][-1]:.3f}  ghost split 平均={r3['Dsplit'].mean():.3e}")
        np.savez("paperconv_test.npz",
                 t=r1['t'], d1=r1['d'], d3=r3['d'], dE1=r1['dE'], dE3=r3['dE'],
                 F2_1=r1['F2'], F2_3=r3['F2'], sqDtD3=r3['sqDtD'], Dsplit3=r3['Dsplit'])

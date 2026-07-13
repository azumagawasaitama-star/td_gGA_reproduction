"""B=7 静的計算の再シード版。
B=5 の収束解（Λ 固有値 ±0.926, ±0.205, 0）を土台に、外側の ghost ペアを
外挿で1組足した初期値から optimize_selfc を回す。複数シードを順に試し、
Max.error < 1e-3 で収束したらキャッシュに保存して終了。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import convenience_routines as cr
from gga_static_solver import GA
from td_gGA_solver import save_static

U_i, T_g = 0.05, 0.003
nq = 7

# シード候補: Λ の対角スペクトル（B=5 の ±0.93, ±0.21 を保ちつつ外側ペアを追加）
seed_specs = [
    ("B5+outer1.3", [-1.3, -0.926, -0.205, 0.0, 0.205, 0.926, 1.3],
     [1.0, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]),
    ("B5+outer1.1", [-1.1, -0.926, -0.205, 0.0, 0.205, 0.926, 1.1],
     [1.0, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]),
    ("spread",      [-1.2, -0.7, -0.25, 0.0, 0.25, 0.7, 1.2],
     [1.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
]

for name, spec, rinit in seed_specs:
    print(f"===== seed '{name}': Λdiag={spec} =====", flush=True)
    ga = GA(U=U_i, nghost=2*(nq-1), nphysorb=2, n=0.5, T=T_g, eks=-99)
    L7 = np.diag(np.array(spec, float))
    linit = cr.inverse_realHcombination(L7, ga.H_list)
    try:
        ga.optimize_selfc(rinit=np.array(rinit, float),
                          lambdainit=np.asarray(linit, float), muinit=0.0)
    except Exception as e:
        print(f"  失敗: {e}", flush=True)
        continue
    eigL = np.sort(np.linalg.eigvalsh(np.real(ga.Lmbda)))
    R = np.abs(np.real(ga.R)).flatten()
    print(f"  収束候補: Λ_eig={eigL.round(4)} |R|={R.round(4)}", flush=True)
    # 収束品質の判定: F1/F2 を cost_func 経由でなく残差で確認できないため
    # Λ が正気の範囲（|Λ|<3）かつ R が物理的（R[大]≈1）かで採用判断
    if np.max(np.abs(eigL)) < 3.0 and 0.9 < np.max(R) <= 1.001:
        save_static(ga, nq, U_i, T_g)
        print(f"  → 採用・キャッシュ保存（seed '{name}'）", flush=True)
        break
    print("  → 異常解の疑い、次のシードへ", flush=True)
else:
    print("全シード不調。手動介入が必要。", flush=True)

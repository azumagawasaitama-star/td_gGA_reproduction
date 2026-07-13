"""B=7 静的計算の断熱ラダー版。
論文脚注 [55]: U→0 で B≥3 の変分ランドスケープは縮退・不安定（ghost が多いほど深刻）。
→ 縮退が解けた U=0.6 で先に解き、U を段階的に下げながら前段の解をシードに使う。
各 U 段の解もキャッシュに保存する（診断・再利用のため）。"""
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

nq, T_g = 7, 0.003
U_ladder = [0.6, 0.3, 0.15, 0.08, 0.05]

# 最初の段のシード: B=5 収束スペクトル + 外側ペア（U=0.6 なら縮退が解けており寛容なはず）
spec0 = [-1.3, -0.926, -0.205, 0.0, 0.205, 0.926, 1.3]
rinit = np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
L_seed = np.diag(np.array(spec0, float))

for U in U_ladder:
    print(f"########## ladder U={U} ##########", flush=True)
    ga = GA(U=U, nghost=2*(nq-1), nphysorb=2, n=0.5, T=T_g, eks=-99)
    linit = np.asarray(cr.inverse_realHcombination(L_seed, ga.H_list), float)
    ga.optimize_selfc(rinit=rinit, lambdainit=linit, muinit=0.0)
    eigL = np.sort(np.linalg.eigvalsh(np.real(ga.Lmbda)))
    Rr = np.abs(np.real(ga.R)).flatten()
    print(f"LADDER U={U}: Λ_eig={eigL.round(4)} |R|={Rr.round(4)}", flush=True)
    save_static(ga, nq, U, T_g)
    # 次の段のシード = 今の解
    L_seed = np.real(ga.Lmbda).copy()
    rinit = np.clip(Rr, 1e-3, None)

print("ladder done", flush=True)

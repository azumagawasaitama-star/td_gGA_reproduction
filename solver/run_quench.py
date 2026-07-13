"""クエンチ実行 CLI（静的キャッシュ利用、結果は data/B{B}/ へ）
使い方:
  python3 run_quench.py --B 3 --dU 1.25 1.5 2.0 2.5 [--Ui 0.05] [--tmax 10] [--bare]
  （--bare: U_f = δU。デフォルトは U_f = U_i + δU）
注意: 異なる B の計算を同一ディレクトリで並走させないこと（演算子 npz が衝突する）。
"""
import argparse
import os
# numpy インポート前にスレッド数を制限すること（OpenBLAS はインポート時に環境変数を読むため、
# numpy を先に import すると後からの os.environ 設定は無効。VECLIB は macOS Accelerate 用）
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from td_gGA_solver import run, get_static, save_run

p = argparse.ArgumentParser()
p.add_argument('--B', type=int, required=True)
p.add_argument('--dU', type=float, nargs='+', required=True)
p.add_argument('--Ui', type=float, default=0.05)
p.add_argument('--T', type=float, default=0.003)
p.add_argument('--tmax', type=float, default=10.0)
p.add_argument('--dt', type=float, default=0.01)
p.add_argument('--Nfreq', type=int, default=60)
p.add_argument('--bare', action='store_true', help='U_f = δU（デフォルト: U_i+δU）')
args = p.parse_args()

ga = get_static(args.B, args.Ui, args.T)
for dU in args.dU:
    U_f = dU if args.bare else args.Ui + dU
    r = run(args.B, U_i=args.Ui, U_f=U_f, T_gGA=args.T, N_freq=args.Nfreq,
            t_max=args.tmax, dt=args.dt, ga=ga, verbose=(args.B >= 5),
            tag=f"B{args.B}-Uf{U_f:g}")
    path = save_run(r, args.B, args.Ui, U_f)
    print(f"B={args.B} U_f={U_f:g}: |ΔE/E|={np.max(r['dE']):.2e} "
          f"F2={np.max(r['F2']):.2e} d_mean={np.mean(r['d']):.4f} → {path}", flush=True)

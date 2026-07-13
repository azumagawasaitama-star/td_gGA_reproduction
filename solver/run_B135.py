"""B = 1, 3, 5 の系統比較（論文 Fig.2a の B 系列再現）。
演算子 npz が B ごとに同名上書きされるため、B を1つずつ完結させて進む。
B=5 は dim_Phi=924: 静的計算が長い（30分〜数時間）。TD は密行列のまま実行。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from td_gGA_solver import run, solve_static

U_i = 0.05
dUs = [1.25, 1.5, 2.0, 2.5]
Bs = [1, 3, 5]

data = {}
for B in Bs:
    t0 = time.time()
    print(f"########## B={B}: 静的計算開始 ##########", flush=True)
    ga = solve_static(B, U_i)
    print(f"########## B={B}: 静的計算 {time.time()-t0:.0f}s ##########", flush=True)
    for dU in dUs:
        t1 = time.time()
        r = run(B, U_i=U_i, U_f=U_i + dU, t_max=10.0, ga=ga,
                verbose=(B == 5), tag=f"B{B}-dU{dU}")
        print(f"  B={B} δU={dU}: |ΔE/E|={r['dE'].max():.2e} F2={r['F2'].max():.2e} "
              f"d_mean={np.mean(r['d']):.4f} ({time.time()-t1:.0f}s)", flush=True)
        for k in ('t', 'd', 'E', 'dE', 'F2', 'sqDtD', 'eigLc'):
            data[f"{k}_B{B}_dU{dU}"] = np.array(r[k])
        np.savez("B135_data.npz", **data)   # 逐次保存（途中で死んでも残る）

print("all done", flush=True)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

paper = np.load('paper_fig2a_digitized.npz')
try:
    p57 = np.load('paper_fig2a_B57_digitized.npz')
except Exception:
    p57 = None

colors = {1: 'gray', 3: 'red', 5: 'blue'}
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
for ax, dU in zip(axes.flat, dUs):
    o = np.argsort(paper[f'tr_{dU}'])
    ax.plot(paper[f'tr_{dU}'][o], paper[f'dr_{dU}'][o], '.', ms=2.5, color='salmon',
            label='paper B=3' if dU == dUs[0] else None)
    if p57 is not None and f't_blue_{dU}' in p57:
        ob = np.argsort(p57[f't_blue_{dU}'])
        ax.plot(p57[f't_blue_{dU}'][ob], p57[f'd_blue_{dU}'][ob], '.', ms=2.5,
                color='cornflowerblue', label='paper B=5' if dU == 2.0 else None)
    for B in Bs:
        ls = '--' if B == 1 else '-'
        ax.plot(data[f"t_B{B}_dU{dU}"], data[f"d_B{B}_dU{dU}"], ls, color=colors[B],
                lw=1.6 if B > 1 else 1.0, label=f'ours B={B}' if dU == dUs[0] else None)
    ax.text(0.03, 0.92, f"$\\delta U={dU}$", transform=ax.transAxes, fontsize=12)
    ax.set_ylim(0, 0.28)
axes[0, 0].legend(fontsize=8, loc='upper right')
for ax in axes[1]:
    ax.set_xlabel('t')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
fig.suptitle('TD-gGA paperconv: B = 1, 3, 5 (vs digitized paper curves)')
plt.tight_layout()
plt.savefig('fig2a_B135.png', dpi=140)
print('saved fig2a_B135.png', flush=True)

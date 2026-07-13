"""論文 Fig.2a の再現: δU = 1.25, 1.5, 2.0, 2.5（U_f = U_i + δU, U_i = 0.05）
B=1 と B=3 を各パネルで比較。静的解は B ごとに1回だけ解いて使い回す。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from td_gGA_solver import run, solve_static

U_i = 0.05
dUs = [1.25, 1.5, 2.0, 2.5]
T_MAX = 10.0

data = {}
for B in (1, 3):
    print(f"===== 静的計算 B={B} (U_i={U_i}) =====", flush=True)
    ga = solve_static(B, U_i)
    for dU in dUs:
        U_f = U_i + dU
        print(f"----- B={B} δU={dU} (U_f={U_f}) -----", flush=True)
        r = run(B, U_i=U_i, U_f=U_f, t_max=T_MAX, ga=ga, tag=f"B{B}-dU{dU}",
                verbose=False)
        print(f"  |ΔE/E|max={r['dE'].max():.2e} F2max={r['F2'].max():.2e} "
              f"d_mean={r['d'].mean():.4f} d_final={r['d'][-1]:.4f}", flush=True)
        data[f"t_B{B}_dU{dU}"] = r['t']
        data[f"d_B{B}_dU{dU}"] = r['d']
        data[f"dE_B{B}_dU{dU}"] = r['dE']
        data[f"F2_B{B}_dU{dU}"] = r['F2']
        data[f"sqDtD_B{B}_dU{dU}"] = r['sqDtD']

np.savez("fig2a_data.npz", **data)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, dU in zip(axes.flat, dUs):
    ax.plot(data[f"t_B1_dU{dU}"], data[f"d_B1_dU{dU}"], '--', color='gray',
            lw=1.5, label='B=1')
    ax.plot(data[f"t_B3_dU{dU}"], data[f"d_B3_dU{dU}"], '-', color='crimson',
            lw=2, label='B=3')
    ax.text(0.05, 0.92, f"$\\delta U = {dU}$", transform=ax.transAxes, fontsize=12)
    ax.set_ylim(0, 0.28); ax.set_xlim(0, T_MAX)
axes[0, 0].legend(loc='upper right')
for ax in axes[1]:
    ax.set_xlabel('t')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
fig.suptitle(r'TD-gGA paperconv: $U_i=0.05 \to U_f=U_i+\delta U$ (cf. PRR 5, L032023 Fig.2a)')
plt.tight_layout()
plt.savefig("fig2a_reproduction.png", dpi=140)
print("saved fig2a_reproduction.png")

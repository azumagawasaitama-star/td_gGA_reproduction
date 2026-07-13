"""論文 Fig.2b 型のプロット（eig Λc と √(D†D)）＋ エネルギー・F2 保存プロット。
B=3, δU = 1.25, 1.5, 2.0, 2.5（U_f = U_i + δU）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from td_gGA_solver import run, solve_static

U_i = 0.05
dUs = [1.25, 1.5, 2.0, 2.5]

ga = solve_static(3, U_i)
runs = {}
for dU in dUs:
    runs[dU] = run(3, U_i=U_i, U_f=U_i + dU, t_max=10.0, ga=ga, verbose=False,
                   tag=f"dU{dU}")
    r = runs[dU]
    print(f"δU={dU}: |ΔE/E|max={r['dE'].max():.2e} F2max={r['F2'].max():.2e}", flush=True)

np.savez("fig2b_data.npz",
         **{f"{k}_dU{dU}": np.array(runs[dU][k])
            for dU in dUs for k in ('t', 'd', 'E', 'dE', 'F2', 'sqDtD', 'eigLc')})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Fig.2b 型: 上段 eig Λc、下段 √(D†D) ---
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for j, dU in enumerate(dUs):
    r = runs[dU]
    t = np.array(r['t']); eig = np.array(r['eigLc'])          # (nt, 3)
    for k in range(eig.shape[1]):
        axes[0, j].plot(t, eig[:, k], color='green', lw=1.0)
    axes[0, j].text(0.05, 0.88, f"$\\delta U = {dU}$", transform=axes[0, j].transAxes)
    axes[0, j].set_ylim(-5.5, 5.5)
    axes[1, j].plot(t, r['sqDtD'], color='green', lw=1.6)
    axes[1, j].set_ylim(0, 0.55)
    axes[1, j].set_xlabel('t')
axes[0, 0].set_ylabel(r'eig $\Lambda^c$')
axes[1, 0].set_ylabel(r'$\sqrt{\mathcal{D}^\dagger \mathcal{D}}$')
fig.suptitle('TD-gGA paperconv B=3: EH パラメータのゲージ不変量（cf. 論文 Fig.2b, B=7）')
plt.tight_layout()
plt.savefig('fig2b_reproduction.png', dpi=140)
print('saved fig2b_reproduction.png')

# --- エネルギー・F2 保存プロット ---
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(dUs)))
for c, dU in zip(colors, dUs):
    r = runs[dU]
    t = np.array(r['t'])
    ax[0].plot(t, r['E'], color=c, lw=1.4, label=f"$\\delta U$={dU}")
    ax[1].semilogy(t, np.maximum(np.array(r['dE']), 1e-16), color=c, lw=1.2)
    ax[2].semilogy(t, np.maximum(np.array(r['F2']), 1e-16), color=c, lw=1.2)
ax[0].set_ylabel(r'$E_{\rm phys}(t)$'); ax[0].set_title('全エネルギー（保存量そのもの）')
ax[1].set_title(r'$|E(t)-E(0)|/|E(0)|$'); ax[1].set_ylim(1e-12, 1e-3)
ax[2].set_title(r'$F_2(t) = \|\langle f^\dagger f\rangle - \int\rho\, n\, d\omega\|$')
ax[2].set_ylim(1e-8, 1e-3)
for a in ax:
    a.set_xlabel('t')
ax[0].legend(fontsize=9)
fig.suptitle('TD-gGA paperconv B=3: エネルギーと F2 拘束の保存（Λ=0 ゲージ、射影なし）')
plt.tight_layout()
plt.savefig('conservation.png', dpi=140)
print('saved conservation.png')

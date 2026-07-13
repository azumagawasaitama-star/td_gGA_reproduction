"""data/ から読んでプロットするだけ（再計算なし）。
使い方: python3 plot_all.py [--Ui 0.05]
data/B{B}/quench_Ui{U_i}_Uf{U_f}.npz を全部拾い、
  - fig2a_B.png       : d(t) パネル（δU 別、B 重ね、論文デジタイズ点つき）
  - fig2b_B.png       : eig Λc / √D†D（B 別の行）
  - conservation_B.png: |ΔE/E| と F2
を生成する。"""
import argparse, glob, os, re
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--Ui', type=float, default=0.05)
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/ はプロジェクトルート直下で共有
runs = {}   # (B, U_f) -> dict
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'B*', 'quench_*.npz'))):
    z = np.load(f)
    if abs(float(z['U_i']) - args.Ui) > 1e-12:
        continue
    runs[(int(z['B']), float(z['U_f']))] = dict(z)
if not runs:
    raise SystemExit('data/ に該当データがありません')

Bs = sorted({k[0] for k in runs})
Ufs = sorted({k[1] for k in runs})
colors = {1: 'gray', 3: 'crimson', 5: 'royalblue', 7: 'seagreen'}
print(f"found: B={Bs}, U_f={Ufs}")

paper = None
try:
    paper = np.load(os.path.join(HERE, 'paper_fig2a_digitized.npz'))
except Exception:
    pass

# --- fig2a: d(t) ---
ncol = len(Ufs)
fig, axes = plt.subplots((ncol + 1) // 2, 2, figsize=(12, 3.8 * ((ncol + 1) // 2)),
                         sharex=True, sharey=True, squeeze=False)
for ax, U_f in zip(axes.flat, Ufs):
    dU_label = U_f - args.Ui
    if paper is not None:
        for dU_p in (1.25, 1.5, 2.0, 2.5):
            if abs(dU_label - dU_p) < 0.06 or abs(U_f - dU_p) < 0.06:
                o = np.argsort(paper[f'tr_{dU_p}'])
                ax.plot(paper[f'tr_{dU_p}'][o], paper[f'dr_{dU_p}'][o], '.', ms=2,
                        color='salmon', alpha=0.7)
    for B in Bs:
        if (B, U_f) not in runs:
            continue
        r = runs[(B, U_f)]
        ax.plot(r['t'], r['d'], '--' if B == 1 else '-', color=colors.get(B, 'k'),
                lw=1.0 if B == 1 else 1.6, label=f'B={B}')
    ax.text(0.03, 0.9, f"$U_f={U_f:g}$", transform=ax.transAxes)
    ax.set_ylim(0, 0.28)
axes[0, 0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('t')
for row in axes:
    row[0].set_ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
plt.tight_layout(); plt.savefig('fig2a_B.png', dpi=140)
print('saved fig2a_B.png')

# --- fig2b: eig Λc / √D†D（B ごとに行、列は U_f）---
Bs_g = [B for B in Bs if B > 1]
if Bs_g:
    fig, axes = plt.subplots(2 * len(Bs_g), ncol, figsize=(3.2 * ncol, 3.0 * 2 * len(Bs_g)),
                             sharex=True, squeeze=False)
    for i, B in enumerate(Bs_g):
        for j, U_f in enumerate(Ufs):
            if (B, U_f) not in runs:
                continue
            r = runs[(B, U_f)]
            eig = r['eigLc']
            for k in range(eig.shape[1]):
                axes[2*i, j].plot(r['t'], eig[:, k], color='green', lw=0.9)
            axes[2*i, j].set_ylim(-6, 6)
            axes[2*i, j].set_title(f"B={B}, $U_f={U_f:g}$", fontsize=9)
            axes[2*i+1, j].plot(r['t'], r['sqDtD'], color='green', lw=1.4)
            axes[2*i+1, j].set_ylim(0, 0.55)
        axes[2*i, 0].set_ylabel(r'eig $\Lambda^c$')
        axes[2*i+1, 0].set_ylabel(r'$\sqrt{D^\dagger D}$')
    for ax in axes[-1]:
        ax.set_xlabel('t')
    plt.tight_layout(); plt.savefig('fig2b_B.png', dpi=140)
    print('saved fig2b_B.png')

# --- 保存則 ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (B, U_f), r in sorted(runs.items()):
    ax[0].semilogy(r['t'], np.maximum(r['dE'], 1e-16), color=colors.get(B, 'k'),
                   lw=1.0, alpha=0.8)
    ax[1].semilogy(r['t'], np.maximum(r['F2'], 1e-16), color=colors.get(B, 'k'),
                   lw=1.0, alpha=0.8)
ax[0].set_title(r'$|E(t)-E(0)|/|E(0)|$'); ax[0].set_ylim(1e-12, 1e-2)
ax[1].set_title(r'$F_2(t)$'); ax[1].set_ylim(1e-8, 1e-2)
for a in ax:
    a.set_xlabel('t')
import matplotlib.lines as mlines
ax[0].legend(handles=[mlines.Line2D([], [], color=colors[B], label=f'B={B}') for B in Bs],
             fontsize=8)
plt.tight_layout(); plt.savefig('conservation_B.png', dpi=140)
print('saved conservation_B.png')

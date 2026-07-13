"""data/ から読んでプロットするだけ（再計算なし）。
1) fig2a_ours_only.png : 我々の結果のみ（B=1,3,5,7 重ね、論文比較なし）— 論文Fig.2a と同一構図
2) compare_B{1,3,5,7}.png : Bごとに我々の結果 vs 論文デジタイズ（同じBのみ）を並べた個別図
   （B=5,7 は論文デジタイズが δU=2.0,2.5 のみのため2パネル、B=1,3 は4パネル）
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/ はプロジェクトルート直下で共有
U_i = 0.05

runs = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'B*', 'quench_*.npz'))):
    z = np.load(f)
    if abs(float(z['U_i']) - U_i) > 1e-12:
        continue
    runs[(int(z['B']), round(float(z['U_f']) - U_i, 2))] = dict(z)  # key = (B, δU)

Bs = sorted({k[0] for k in runs})
dUs = sorted({k[1] for k in runs})
colors = {1: 'gray', 3: 'crimson', 5: 'royalblue', 7: 'seagreen'}

paper1 = np.load(os.path.join(HERE, 'paper_fig2a_digitized.npz'))       # B=1(gray), B=3(red)
paper2 = np.load(os.path.join(HERE, 'paper_fig2a_B57_digitized.npz'))  # B=5(blue), B=7(green)

# ---------- 1) 我々の結果だけ（論文比較なし） ----------
ncol = len(dUs)
fig, axes = plt.subplots((ncol + 1) // 2, 2, figsize=(12, 3.8 * ((ncol + 1) // 2)),
                         sharex=True, sharey=True, squeeze=False)
for ax, dU in zip(axes.flat, dUs):
    for B in Bs:
        if (B, dU) not in runs:
            continue
        r = runs[(B, dU)]
        ax.plot(r['t'], r['d'], '--' if B == 1 else '-', color=colors.get(B, 'k'),
                lw=1.2 if B == 1 else 1.8, label=f'B={B}')
    ax.text(0.03, 0.9, f"$\\delta U={dU:g}$", transform=ax.transAxes, fontsize=11)
    ax.set_ylim(0, 0.28)
axes[0, 0].legend(fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel('t')
for row in axes:
    row[0].set_ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
fig.suptitle('TD-gGA paperconv: our results only (no paper overlay)')
plt.tight_layout()
plt.savefig('fig2a_ours_only.png', dpi=140)
print('saved fig2a_ours_only.png')

# ---------- 2) B ごとの個別比較（我々 vs 論文デジタイズ、同じBのみ） ----------
digitized_map = {
    # 注意: 保存キーは f'{dU}' 形式（2.0 は "2.0" のまま）。:g だと "2" になり不一致で
    # パネルが欠落するので、ここは :g を使わないこと。
    1: lambda dU: (paper1[f'tg_{dU}'], paper1[f'dg_{dU}']) if f'tg_{dU}' in paper1.files else None,
    3: lambda dU: (paper1[f'tr_{dU}'], paper1[f'dr_{dU}']) if f'tr_{dU}' in paper1.files else None,
    5: lambda dU: (paper2[f't_blue_{dU}'], paper2[f'd_blue_{dU}']) if f't_blue_{dU}' in paper2.files else None,
    7: lambda dU: (paper2[f't_green_{dU}'], paper2[f'd_green_{dU}']) if f't_green_{dU}' in paper2.files else None,
}
dot_colors = {1: '0.5', 3: 'salmon', 5: 'cornflowerblue', 7: 'lightgreen'}

for B in Bs:
    avail_dUs = [dU for dU in dUs if digitized_map[B](dU) is not None and (B, dU) in runs]
    if not avail_dUs:
        print(f'B={B}: 論文デジタイズなし、スキップ')
        continue
    n = len(avail_dUs)
    ncol_b = min(n, 2)
    nrow_b = (n + ncol_b - 1) // ncol_b
    fig, axes = plt.subplots(nrow_b, ncol_b, figsize=(6.5 * ncol_b, 4.2 * nrow_b),
                             squeeze=False)
    for ax, dU in zip(axes.flat, avail_dUs):
        tp, dp = digitized_map[B](dU)
        o = np.argsort(tp)
        ax.plot(tp[o], dp[o], '.', ms=3, color=dot_colors[B], label='paper (digitized)')
        r = runs[(B, dU)]
        ax.plot(r['t'], r['d'], '-', color=colors[B], lw=1.8, label=f'ours B={B}')
        ax.text(0.03, 0.9, f"$\\delta U={dU:g}$", transform=ax.transAxes, fontsize=11)
        ax.set_ylim(0, 0.28); ax.set_xlabel('t')
        ax.set_ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
        ax.legend(fontsize=9)
    for k in range(len(avail_dUs), nrow_b * ncol_b):
        axes.flat[k].axis('off')
    fig.suptitle(f'B={B}: ours vs paper (digitized)')
    plt.tight_layout()
    fname = f'compare_B{B}.png'
    plt.savefig(fname, dpi=140)
    print('saved', fname)

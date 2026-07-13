"""論文 Fig.2a をデジタイズし、我々の B=1/B=3（plus: U_f=U_i+δU / bare: U_f=δU）と
直接重ねて定量比較する。PPT 目視の代替（軸校正は目盛りマークで厳密）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from PIL import Image

PAGE = '/private/tmp/claude-501/-Users-azumagawayoshito/1b33c511-04d1-4816-9859-e001119d0d92/scratchpad/page4-4.png'
im = np.array(Image.open(PAGE).convert('RGB')).astype(int)
panels = {1.25: (355, 1276, 1406, 1724), 1.5: (1381, 2302, 1406, 1724),
          2.0: (355, 1276, 1793, 2112), 2.5: (1381, 2302, 1793, 2112)}

def xticks_map(x0, x1, y1):
    band = (im[y1+1:y1+13, x0-8:x1+9].max(axis=2) < 120)
    cols = np.where(band.sum(axis=0) >= 3)[0]
    groups = []
    for c in cols:
        if groups and c - groups[-1][-1] <= 4: groups[-1].append(c)
        else: groups.append([c])
    xt = np.array([np.mean(g) for g in groups]) + (x0-8)
    sp = np.median(np.diff(xt))
    idx = np.round((xt - xt[0]) / sp)
    A = np.polyfit(xt, 2.0*idx, 1)
    return lambda px: A[0]*px + A[1]

def ytick_spacing(x0, y0, y1):
    band = (im[y0-5:y1+5, x0-10:x0-1].max(axis=2) < 120)
    rows = np.where(band.sum(axis=1) >= 2)[0]
    groups = []
    for rr in rows:
        if groups and rr - groups[-1][-1] <= 4: groups[-1].append(rr)
        else: groups.append([rr])
    cent = np.array([np.mean(g) for g in groups])
    dif = np.diff(cent)
    dy = np.median(dif)
    assert len(cent) >= 3 and np.abs(dif - dy).max() < 4, f"yticks bad: {cent}"
    return dy

def extract(crop, mask):
    xs, ys = [], []
    for x in range(crop.shape[1]):
        yy = np.where(mask[:, x])[0]
        if len(yy) >= 2:
            xs.append(x); ys.append(np.median(yy))
    return np.array(xs, float), np.array(ys, float)

paper = {}
for dU, (x0, x1, y0, y1) in panels.items():
    to_t = xticks_map(x0, x1, y1)
    dy = ytick_spacing(x0, y0, y1)
    crop = im[y0+3:y1-2, x0+3:x1-2]
    r, g, b = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    gray = (abs(r-g)<25) & (abs(g-b)<25) & (r>150) & (r<225)
    red = (r>150) & (g<100) & (b<100)
    if dU == 1.25:
        gray[:220, 690:] = False; red[:220, 690:] = False
    xs_g, ys_g = extract(crop, gray)
    xs_r, ys_r = extract(crop, red)
    scale = 0.05 / dy
    to_d = lambda ypx: 0.2463 - (ypx - ys_g.min()) * scale   # 破線最大 = d(0)
    paper[dU] = dict(tg=to_t(xs_g + x0 + 3), dg=to_d(ys_g),
                     tr=to_t(xs_r + x0 + 3), dr=to_d(ys_r))
    print(f"δU={dU}: dy={dy:.1f}px  gray d∈[{to_d(ys_g.max()):.3f},0.246]  "
          f"red d∈[{to_d(ys_r.max()):.3f},{to_d(ys_r.min()):.3f}]", flush=True)

np.savez('paper_fig2a_digitized.npz',
         **{f"{k}_{dU}": v for dU, dd in paper.items() for k, v in dd.items()})

# --- 我々の曲線: plus は fig2a_data.npz、bare は再計算して保存 ---
from td_gGA_solver import run, solve_static
plus = np.load('fig2a_data.npz')
bare = {}
for B in (1, 3):
    ga = solve_static(B, 0.05)
    for dU in panels:
        rr = run(B, U_i=0.05, U_f=dU, t_max=10.0, ga=ga, verbose=False, tag=f"bare-B{B}-{dU}")
        bare[(B, dU)] = (rr['t'], rr['d'])
        print(f"  bare B={B} δU={dU} done", flush=True)
np.savez('fig2a_bare_data.npz', **{f"t_B{B}_dU{dU}": v[0] for (B, dU), v in bare.items()},
         **{f"d_B{B}_dU{dU}": v[1] for (B, dU), v in bare.items()})

# --- プロット: 論文(デジタイズ) vs 我々(plus/bare) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True, sharey=True)
for ax, dU in zip(axes.flat, panels):
    P = paper[dU]
    o = np.argsort(P['tr'])
    ax.plot(P['tg'][np.argsort(P['tg'])], P['dg'][np.argsort(P['tg'])], '.', ms=2,
            color='0.75', label='論文 B=1 (digitized)')
    ax.plot(P['tr'][o], P['dr'][o], '.', ms=3, color='salmon', label='論文 B=3 (digitized)')
    ax.plot(plus[f"t_B1_dU{dU}"], plus[f"d_B1_dU{dU}"], '--', color='0.4', lw=1)
    ax.plot(*bare[(1, dU)], ':', color='0.4', lw=1)
    ax.plot(plus[f"t_B3_dU{dU}"], plus[f"d_B3_dU{dU}"], '-', color='crimson', lw=1.6,
            label='我々 B=3 plus (U_f=U_i+δU)')
    ax.plot(*bare[(3, dU)], '-', color='darkblue', lw=1.6, label='我々 B=3 bare (U_f=δU)')
    ax.text(0.03, 0.92, f"$\\delta U={dU}$", transform=ax.transAxes, fontsize=12)
    ax.set_ylim(0.0, 0.28)
axes[0, 0].legend(fontsize=8, loc='upper right')
for ax in axes[1]:
    ax.set_xlabel('t')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
fig.suptitle('論文 Fig.2a デジタイズ vs 我々の TD-gGA（B=1: 破線/点線、B=3: 実線）')
plt.tight_layout()
plt.savefig('digitized_comparison.png', dpi=140)
print('saved digitized_comparison.png')

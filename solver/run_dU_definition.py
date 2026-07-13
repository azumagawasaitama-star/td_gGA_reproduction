"""δU の定義の切り分け: U_f = U_i + δU vs U_f = δU
B=1 の振動周期（厳密力学）が判別器。両定義で B=1/B=3 を計算し、
ピーク時刻の表と重ね比較用の4パネル図を出力する。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy.signal import find_peaks
from td_gGA_solver import run, solve_static

U_i = 0.05
dUs = [1.25, 1.5, 2.0, 2.5]

def peaks_of(t, d):
    idx, _ = find_peaks(d, prominence=0.005)
    return t[idx]

# 演算子 npz は B ごとに同名で上書きされるため、B の外側ループで順に処理する
runs = {}
for B in (1, 3):
    ga = solve_static(B, U_i)
    for dU in dUs:
        for label, U_f in [("plus", U_i + dU), ("bare", dU)]:
            runs[(B, label, dU)] = run(B, U_i=U_i, U_f=U_f, t_max=10.0, ga=ga,
                                       verbose=False, tag=f"B{B}-{label}{dU}")
            print(f"  done B={B} {label} δU={dU}", flush=True)

data = {(label, dU): (runs[(1, label, dU)], runs[(3, label, dU)])
        for dU in dUs for label in ("plus", "bare")}

print("\n=== B=1 ピーク時刻（破線の山）===")
print(f"{'δU':>5} | {'U_f=U_i+δU の山':>28} | {'U_f=δU の山':>28}")
for dU in dUs:
    p_plus = np.round(peaks_of(runs[(1, 'plus', dU)]['t'], runs[(1, 'plus', dU)]['d']), 2)
    p_bare = np.round(peaks_of(runs[(1, 'bare', dU)]['t'], runs[(1, 'bare', dU)]['d']), 2)
    print(f"{dU:>5} | {str(p_plus):>28} | {str(p_bare):>28}", flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for label, fname, title in [
        ("plus", "fig2a_Uf_plus.png", r"$U_f = U_i + \delta U$"),
        ("bare", "fig2a_Uf_bare.png", r"$U_f = \delta U$")]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, dU in zip(axes.flat, dUs):
        r1, r3 = data[(label, dU)]
        ax.plot(r1['t'], r1['d'], '--', color='gray', lw=1.5, label='B=1')
        ax.plot(r3['t'], r3['d'], '-', color='crimson', lw=2, label='B=3')
        ax.text(0.05, 0.92, f"$\\delta U = {dU}$", transform=ax.transAxes, fontsize=12)
        ax.set_ylim(0, 0.28); ax.set_xlim(0, 10)
    axes[0, 0].legend(loc='upper right')
    for ax in axes[1]:
        ax.set_xlabel('t')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
    fig.suptitle(f"TD-gGA paperconv: {title}")
    plt.tight_layout(); plt.savefig(fname, dpi=140)
    print("saved", fname)

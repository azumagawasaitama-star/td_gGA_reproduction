"""U_i=0.05, B=3 の静的鞍点ファミリー探索。
シードを振って optimize_selfc を収束させ、異なる鞍点ごとに δU=2.0 の TD を実行し、
論文 Fig.2a の赤（デジタイズ済み paper_fig2a_digitized.npz）と照合する。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from gga_static_solver import GA
from td_gGA_solver import run

U_i, dU = 0.05, 2.0
U_f = U_i + dU     # plus 規約（δU=1.25 で論文と一致が良かった方）

# H_list 順: [E11, E22, E33, (12), (13), (23)]
seeds = [
    ("baseline",    [1, 0, 0],       [1, 0, 0, 0, 0, 0]),
    ("ghost-split", [1, 0.3, 0.3],   [0, 1, -1, 0, 0, 0]),
    ("mixed-1",     [0.8, 0.4, 0.4], [0.5, 1, -1, 0, 0, 0]),
    ("offdiag-1",   [1, 0.2, -0.2],  [0, 0.8, -0.8, 0.3, 0.3, 0]),
    ("offdiag-2",   [1, 0.5, 0.5],   [0, 0, 0, 0.7, 0.7, 0]),
    ("mixed-2",     [0.7, 0.5, -0.5],[1, 0.5, -0.5, 0, 0, 0.5]),
    ("bigL-1",      [1, 0.1, 0.1],   [0, 1.5, -1.5, 0, 0, 0]),
    ("bigL-2",      [1, 0.05, 0.05], [0, 2, -2, 0, 0, 0]),
]

paper = np.load('paper_fig2a_digitized.npz')
tp, dp = paper[f'tr_{dU}'], paper[f'dr_{dU}']
o = np.argsort(tp); tp, dp = tp[o], dp[o]

results = {}
seen = []
for name, rs, ls in seeds:
    print(f"===== seed '{name}' =====", flush=True)
    ga = GA(U=U_i, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)
    try:
        ga.optimize_selfc(rinit=np.array(rs, float), lambdainit=np.array(ls, float),
                          muinit=0.0)
    except Exception as e:
        print(f"  収束失敗: {e}", flush=True)
        continue
    eigL = np.sort(np.linalg.eigvalsh(np.real(ga.Lmbda)))
    R_imp = np.abs(np.real(ga.R)).flatten()
    print(f"  Λ_eig={eigL.round(4)}  |R|={R_imp.round(4)}", flush=True)
    key = tuple(eigL.round(2))
    if any(np.allclose(key, k, atol=0.02) for k in seen):
        print("  → 既出の鞍点、スキップ", flush=True)
        continue
    seen.append(key)
    try:
        r = run(3, U_i=U_i, U_f=U_f, t_max=10.0, ga=ga, verbose=False, tag=name)
    except Exception as e:
        print(f"  TD 失敗: {e}", flush=True)
        continue
    di = np.interp(tp, r['t'], r['d'])
    rms = float(np.sqrt(np.mean((di - dp) ** 2)))
    m = (r['t'] > 3) & (r['t'] < 7)
    ipk = np.argmax(r['d'][m])
    print(f"  TD: |ΔE/E|={r['dE'].max():.1e} F2={r['F2'].max():.1e} "
          f"RMS(論文赤)={rms:.4f} revival=(t={r['t'][m][ipk]:.2f}, d={r['d'][m][ipk]:.3f})",
          flush=True)
    results[name] = dict(t=r['t'], d=r['d'], rms=rms, eigL=eigL)

print("\n=== まとめ（RMS 昇順; 論文 revival ≈ (t=4.8, d=0.163)）===")
for name in sorted(results, key=lambda n: results[n]['rms']):
    v = results[name]
    print(f"  {name:12s} RMS={v['rms']:.4f}  Λ={v['eigL'].round(3)}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5.5))
plt.plot(tp, dp, '.', ms=4, color='salmon', label='論文 B=3 (digitized)')
for name, v in results.items():
    plt.plot(v['t'], v['d'], lw=1.4, label=f"{name} (RMS={v['rms']:.3f})")
plt.xlabel('t'); plt.ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
plt.title(f'鞍点ファミリー vs 論文赤 (B=3, U={U_i}→{U_f})')
plt.legend(fontsize=8); plt.ylim(0, 0.28)
plt.tight_layout(); plt.savefig('saddle_scan.png', dpi=140)
print('saved saddle_scan.png')
np.savez('saddle_scan_data.npz',
         **{f"{n}//{k}": v[k] for n, v in results.items() for k in ('t', 'd')})

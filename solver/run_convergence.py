"""数値設定の感度チェック（B=3, δU=2.5）: N_freq, dt, T のどれが d(t) を動かすか"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from td_gGA_solver import run, solve_static

U_i, dU = 0.05, 2.5
U_f = U_i + dU
cases = [
    ("base (Nf=60, dt=0.01, T=3e-3)", dict(N_freq=60,  dt=0.01,  T_gGA=0.003)),
    ("Nf=150",                        dict(N_freq=150, dt=0.01,  T_gGA=0.003)),
    ("Nf=300",                        dict(N_freq=300, dt=0.01,  T_gGA=0.003)),
    ("dt=0.005",                      dict(N_freq=60,  dt=0.005, T_gGA=0.003)),
    ("T=1e-3",                        dict(N_freq=60,  dt=0.01,  T_gGA=0.001)),
]

ga_cache = {}
data = {}
for label, kw in cases:
    T_g = kw['T_gGA']
    if T_g not in ga_cache:
        print(f"===== 静的計算 B=3 (T={T_g}) =====", flush=True)
        ga_cache[T_g] = solve_static(3, U_i, T_gGA=T_g)
    print(f"----- {label} -----", flush=True)
    r = run(3, U_i=U_i, U_f=U_f, t_max=10.0, ga=ga_cache[T_g], verbose=False,
            tag=label, **kw)
    print(f"  |ΔE/E|max={r['dE'].max():.2e} F2max={r['F2'].max():.2e} "
          f"d_mean={r['d'].mean():.4f}", flush=True)
    data[label] = (r['t'], r['d'])

base_t, base_d = data[cases[0][0]]
print("\n=== base との最大差 ===")
for label, (t, d) in data.items():
    if label == cases[0][0]:
        continue
    di = np.interp(base_t, t, d)
    print(f"  {label}: max|Δd| = {np.abs(di - base_d).max():.3e}")

np.savez("convergence_data.npz", **{f"{k}//t": v[0] for k, v in data.items()},
         **{f"{k}//d": v[1] for k, v in data.items()})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))
for label, (t, d) in data.items():
    lw = 2.5 if label.startswith('base') else 1.2
    plt.plot(t, d, lw=lw, label=label)
plt.xlabel('t'); plt.ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
plt.title(f'B=3, $\\delta U$={dU} 設定感度チェック')
plt.legend(); plt.ylim(0, 0.28)
plt.tight_layout(); plt.savefig('convergence_check.png', dpi=140)
print("saved convergence_check.png")

# TD-gGA 論文再現パッケージ

Guerci, Capone, Lanatà, *Phys. Rev. Research* **5**, L032023 (2023) の
時間依存ゴースト・グッツヴィラー近似（TD-gGA）を実装し、論文 Fig.2a/Fig.2b の
クエンチダイナミクスを B=1,3,5,7 で再現したもの。

DMFT（数値厳密）を審判にした比較では、本実装の B=3 の方が論文本文の B=3 曲線より
DMFT に近いという結果も得ている（詳細は `design_doc.pdf` §7 以降 / `POSTMORTEM_2026-07-07.md`）。

## 必要環境

```
python3, numpy, scipy, matplotlib, primme (pip install primme)
```

`primme`（疎行列固有値ソルバー）は `ed_solver.py` が起動時に import するため、
B=1 のみ動かす場合でも必須。検証用の `compare_digitized.py` を使う場合のみ
追加で `Pillow`（`pip install Pillow`）が要る。

**numpy スレッド数の注意**: 各スクリプトは `import numpy` の**前**に
`OPENBLAS_NUM_THREADS` 等を `1` に固定している。これは並列実行時の速度低下
（実測1000倍以上）を防ぐための必須の作法。新しいスクリプトを書く場合もこの順序を守ること。

## クイックスタート

```bash
cd solver
python3 plot_all.py            # 同梱済みのキャッシュ済み結果からFig.2a/2b相当の図を再生成（数秒、計算なし）
```

`static_cache/`・`data/` に既に計算済みの結果を同梱しているので、上のコマンドだけで
`fig2a_B.png` / `fig2b_B.png` / `conservation_B.png` が `solver/` 直下に再生成される。
`reference_figures/` に同じ図の完成版を置いてあるので見比べられる。

**ゼロから計算し直したい場合**（結果は同じになるはず。B=5 で数分〜数十分、
B=7 は静的計算のシード探索を含め数十分〜）:

```bash
cd solver
python3 run_fig2a.py           # B=1, B=3 の4パネル(δU=1.25,1.5,2.0,2.5)を計算・プロット
python3 run_fig2b.py           # Fig.2b 相当（eig Λc, √D†D）
python3 run_B135.py            # B=1,3,5 系統比較
python3 b7_ladder.py           # B=7 静的解を断熱ラダーで求める（先にこれを走らせる）
python3 b7_reseed.py           # ↑がうまくいかない場合の代替シード探索
python3 run_quench.py --B 3 --dU 1.25 1.5 2.0 2.5   # 任意のB・クエンチ幅を直接指定するCLI
```

**注意**: 演算子 `.npz`（`H1mat-*.npz` 等）は実行時に `solver/` 直下へ自動生成される。
**異なる B の計算を同じフォルダで同時に走らせないこと**（ファイルが衝突する）。

## 構成

```
td_gGA_reproduce/
├── README.md                              このファイル
├── design_doc.pdf / design_doc.tex        設計図（全体像・物理・運用ルール）
├── derivation_F2_conservation_proof.md    理論: 複素TDVPの一貫規約とF2厳密保存の証明
├── POSTMORTEM_2026-07-07.md               「何が問題だったか」の総括（試行錯誤の教訓）
├── static_cache/                          静的鞍点の計算済みキャッシュ(B=1,3,5,7)
├── data/                                  TDクエンチ結果(B=1,3,5,7 × δU=1.25,1.5,2.0,2.5)
├── reference_figures/                     完成版の図（見比べ用）
└── solver/                                コード一式（下記）
    ├── td_gGA_solver.py                    ★メインの時間発展ソルバー（run/solve_static等）
    ├── tdvp_core.py                        ソルバーが使う基礎ヘルパー(H_emb構築・pack/unpack等)
    ├── tdvp_sparse.py                      同上、疎行列版ヘルパー(B=5,7で使用)
    ├── gga_static_solver.py                静的gGAソルバー(GAクラス、クエンチ前の平衡状態を解く)
    ├── ed_solver.py                        埋め込みハミルトニアンの厳密対角化
    ├── convenience_routines.py             汎用の行列補助関数
    ├── lattice.py                          半円形DOS等の格子・バス設定
    ├── run_fig2a.py / run_fig2b.py         論文Fig.2a/2b 再現ランナー
    ├── run_B135.py                         B=1,3,5 系統比較ランナー
    ├── run_quench.py                       任意条件のクエンチCLI
    ├── b7_ladder.py / b7_reseed.py         B=7 静的解の断熱シード探索
    ├── plot_all.py / plot_compare_perB.py  data/ から図を再生成するだけ(再計算なし)
    ├── saddle_scan.py                      静的鞍点の多重性チェック(検証用)
    ├── run_convergence.py                  数値設定(N_freq,dt,T)の感度チェック(検証用)
    ├── run_dU_definition.py                δU定義(plus/bare)の切り分け(検証用)
    ├── compare_digitized.py                論文PDFのデジタイズ比較(検証用、要手動パス設定)
    └── paper_fig2a_digitized.npz 等        論文Fig.2aのデジタイズ済み比較データ
```

上記のうち `saddle_scan.py` / `run_convergence.py` / `run_dU_definition.py` /
`compare_digitized.py` の4本は「論文の主結果を再現する」ためには不要で、
再現結果の信頼性を検証した際の副産物（数値設定を振っても結果が変わらないことの
確認、静的鞍点が一意であることの確認など）。`compare_digitized.py` は論文PDFの
ページ画像から曲線を読み取るためのスクリプトで、そのままでは動かない
（`PAGE`変数に自分で用意した論文図のPNGパスを設定する必要がある）。

## 補足: ファイル名について

元の開発リポジトリでは `ga_mainfin_routeA.py` / `td_gGA_solver_routeA.py` /
`td_gGA_solver_routeC.py` / `td_gGA_solver_paperconv.py` という、試行錯誤の過程
（Route A, Route C など）を反映した名前が付いていた。このパッケージでは
役割がわかる名前に変更している：

| 旧名（開発リポジトリ） | 新名（本パッケージ） | 役割 |
|---|---|---|
| `td_gGA_solver_paperconv.py` | `td_gGA_solver.py` | 実際に使う、正しい時間発展ソルバー |
| `td_gGA_solver_routeA.py` | `tdvp_core.py` | ヘルパー関数群（自身の時間発展ドライバは未使用） |
| `td_gGA_solver_routeC.py` | `tdvp_sparse.py` | ヘルパー関数群・疎行列版（自身の時間発展ドライバは未使用） |
| `ga_mainfin_routeA.py` | `gga_static_solver.py` | 静的解（クエンチ前の平衡状態）を解く |

`tdvp_core.py` / `tdvp_sparse.py` はファイル内に自分自身の時間発展ドライバも
含んでいるが、規約バグ（B≥3でエネルギー非保存等）があり**使われていない**。
実際に使われているのは `td_gGA_solver.py` から import されるヘルパー関数のみ
（各ファイル冒頭に注記あり）。詳しい経緯は `POSTMORTEM_2026-07-07.md` を参照。

## 到達している精度（既定セッティング）

| B | ノークエンチ \|ΔE/E\| | クエンチ \|ΔE/E\| | F2 |
|---|---|---|---|
| 1 | 機械精度 | 機械精度 | 機械精度 |
| 3 | ~1e-6 | ~1e-7 | ~1e-6（一定） |
| 5,7 | 同程度（適応刻み積分器 DOP853 使用時） | | |

詳細な数値・導出は `design_doc.pdf` §7「数値セッティング一覧」を参照。

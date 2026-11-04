"""validate_c_outputs.py のCSVから比較図と日本語レポートを生成する。"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "validation_c_5000_20261027"


def main():
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    with open(ROOT / "comparison_by_step.csv", encoding="utf-8", newline="") as stream:
        comparisons = list(csv.DictReader(stream))
    with open(ROOT / "field_statistics.csv", encoding="utf-8", newline="") as stream:
        stats = list(csv.DictReader(stream))

    def select_stats(implementation, field):
        return sorted((row for row in stats if row["implementation"] == implementation and row["field"] == field),
                      key=lambda row: int(row["step"]))

    def selected_comparisons(variant, field):
        return sorted((row for row in comparisons if row["variant"] == variant and row["field"] == field),
                      key=lambda row: int(row["step"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"python": "#176bb0", "original_c": "#777777", "matched_c": "#d65f24"}
    labels = {"python": "Python (saved user run)", "original_c": "C original settings", "matched_c": "C matched settings"}
    for implementation in colors:
        ne = select_stats(implementation, "ne")
        valid = [r for r in ne if int(r["negative"]) == 0 and int(r["nan"]) == 0 and int(r["inf"]) == 0
                 and r["max"] and 0 < float(r["max"]) < 1e30]
        axes[0, 0].plot([int(r["step"]) * .001 for r in valid], [float(r["max"]) for r in valid],
                        label=labels[implementation], color=colors[implementation],
                        linestyle="--" if implementation == "matched_c" else "-")
        field = select_stats(implementation, "absE")
        valid = [r for r in field if int(r["nan"]) == 0 and int(r["inf"]) == 0 and r["max"] and float(r["max"]) < 1e5]
        axes[0, 1].plot([int(r["step"]) * .001 for r in valid], [float(r["max"]) for r in valid],
                        label=labels[implementation], color=colors[implementation],
                        linestyle="--" if implementation == "matched_c" else "-")
    axes[0, 0].set(yscale="log", xlabel="Step label time [ns]", ylabel="Maximum electron density [m$^{-3}$]",
                   title="Electron density before numerical breakdown")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel="Step label time [ns]", ylabel="Maximum reduced field [Td]",
                   title="Electric field (finite, bounded frames)")
    axes[0, 1].legend(fontsize=8)
    for field in ("ne", "phi", "absE", "Ex", "Ey"):
        rows = selected_comparisons("matched_c", field)
        finite = [r for r in rows if int(r["step"]) <= 4800 and r["scaled_max_error"]]
        axes[1, 0].plot([int(r["step"]) * .001 for r in finite],
                        [max(float(r["scaled_max_error"]), 1e-16) for r in finite], label=field)
    axes[1, 0].set(yscale="log", xlabel="Step label time [ns]", ylabel="max |Python - C| / max |C|",
                   title="Python versus matched C: steps 0-4800\nExact equality shown at 1e-16", xlim=(0, 5))
    axes[1, 0].legend(fontsize=8)
    for implementation in colors:
        ne = select_stats(implementation, "ne")
        axes[1, 1].plot([int(r["step"]) * .001 for r in ne],
                        [int(r["negative"]) / 20000 for r in ne],
                        label=labels[implementation] + ": ne < 0", color=colors[implementation])
        field = select_stats(implementation, "absE")
        axes[1, 1].plot([int(r["step"]) * .001 for r in field],
                        [(int(r["nan"]) + int(r["inf"])) / 20000 for r in field],
                        label=labels[implementation] + ": nonfinite field", color=colors[implementation], linestyle=":")
    axes[1, 1].set(xlabel="Step label time [ns]", ylabel="Fraction of cells", ylim=(-.03, 1.03),
                   title="Numerical validity of the saved output")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(ROOT / "comparison_overview.png", dpi=160)
    fig.savefig(ROOT / "comparison_overview.pdf")
    plt.close(fig)

    lines = [
        "# Python 5000 step出力とC版の実行比較", "",
        "## 判定", "",
        "**今回のPython出力は5000 stepまでの正常な数値計算としては不合格です。**",
        "step 4900以降に負の電子密度とInfの換算電場が記録されています。計算処理が終了していても、この区間の結果は利用できません。", "",
        f"Pythonと条件を合わせたC版のstep 4800までの最大スケール誤差は "
        f"`{summary['cases']['matched_c']['max_scaled_error_through_4800']:.6e}` でした。",
        "C版との数値一致と、計算が安定しているかどうかを別々に評価しています。", "",
        "元Cをそのまま実行する条件と、今回のPython計算条件には背景電離・Poisson解法・収束判定などの差があります。",
        "元Cの結果との不一致を移植ミスと取り違えないため、2種類のC実行で比較しました。", "",
        "## 対象", "",
        "検証対象: `streamer_python/outputdata_notebook` の257 CSV。step 0～5000、100 step間隔、各物理量200×100セルです。",
        "今回の実行はstep 0を含む5001回更新、時刻ラベル0～5 nsです。既存Python出力、元Cコード、元C出力は変更していません。", "",
        "## 比較条件", "",
        "Notebookに保存された設定出力を確認しました。共通条件は200×100セル、格子幅10 µm、DC 4 kV、dt=1 ps、",
        "初期Gaussian振幅1e19 m⁻³、z0=900 µm、幅sr0=sz0=100 µm、電極曲率300 µm、Poisson更新量の許容値1e-6です。", "",
        "| 項目 | 保存済みPython出力 | 元Cコードの新規実行 | 条件合わせCの新規実行 |",
        "|---|---|---|---|",
        "| 背景電離 [m⁻³/更新] | 1e13 | 1e14 | 1e13 |",
        "| Poisson更新 | SOR、omega=1.5 | Gauss–Seidel、omega=1 | SOR、omega=1.5 |",
        "| 収束判定の電位尺度 | 最大絶対値 | 符号付き値を保存 | 最大絶対値 |",
        "| 初期空間電荷電位pphi | ゼロ | malloc後に明示初期化なし | callocでゼロ |", "",
        "`original_c/source/` は元Cソース・ヘッダーのコピーで、変更せずコンパイルしました。",
        "`matched_c/source/` だけに上表の差を合わせる変更を加えています。変更差分は `matched_c/changes.diff` に保存しました。",
        "物理的な反応式、移流式、格子、時間刻みは変更していません。", "",
        "最終検証は `stream_wrapper.c` を使い、CSV用のfopen/fprintf/fcloseだけを標準出力パイプに置き換えました。",
        "Cの計算処理は同じソースをincludeして実行し、Python側で受信したCSVを各ファイル相当ごとに即時比較しています。",
        "元ソースの `void main` はラッパーから呼び、ラッパーの標準的な `int main` が終了コード0を返します。", "",
        "コンパイラはローカルのClang 20.1.8（Windows x64）です。`-O2 -fno-fast-math -ffp-contract=off` でビルドしました。",
        "元の `streamer.exe` はCygwin依存のため使用せず、提示されたCソースを新規ビルドして実行しました。",
        "2つのC実行は並行実行したため、記録した経過時間は言語間の速度比較には使えません。", "",
        "## CSV比較", "",
        "CSVの数値を読み戻して各step・各セルを比較しました。WindowsネイティブCの改行CRLFとPythonのLFは区別して記録しています。",
        "`comparison_by_step.csv` に5物理量×51時刻×2条件の510行、`field_statistics.csv` に有限値・負値の診断を保存しました。", "",
        "| C実行 | step 0～4800で有限値かつ全要素一致したCSV | 同じ区間でLF改行に統一すると完全一致 | 同じ区間の最大スケール誤差 |",
        "|---|---:|---:|---:|",
    ]
    for variant, label in (("original_c", "元C設定"), ("matched_c", "Python条件合わせC")):
        result = summary["cases"][variant]
        lines.append(f"| {label} | {result['through_4800_value_equal']} / {result['through_4800_files']} | "
                     f"{result['through_4800_lf_normalized_equal']} / {result['through_4800_files']} | {result['max_scaled_error_through_4800']:.6e} |")
    prior = summary.get("original_c_vs_existing_output")
    if prior:
        lines += ["", f"元Cの新規実行と既存の `streamer/outputdata` も比較しました。"
                  f"{prior['files']}ファイル中{prior['value_equal']}ファイルで全数値が一致しました。"]
    lines += ["", "条件合わせCでは、電子密度ne・電位phi・電場absE/Eyの49時刻分はすべてCSV数値が一致しました。",
              "差があった3ファイルはいずれも半径方向電場Exで、最大絶対差はstep 4300が1e-11 Td、",
              "step 4700が3e-10 Td、step 4800が1e-7 Tdでした。"]
    lines += ["", "スケール誤差 = `max(abs(Python-C))/max(abs(C))`。非有限値を含む比較は誤差評価の対象外とし、",
              "NaN/Infの個数を別記しました。ゼロ近傍の局所相対誤差と混同しないため、分母は配列全体の最大絶対値です。", "",
              "## 主な時刻の値", "",
              "| 実装 | step | 最小ne [m⁻³] | 最大ne [m⁻³] | 負のneのセル数 | absEのNaN/Infセル数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for step in (0, 1000, 2800, 4000, 4800, 4900, 5000):
        for implementation, label in (("python", "Python"), ("original_c", "元C"), ("matched_c", "条件合わせC")):
            ne = next(r for r in select_stats(implementation, "ne") if int(r["step"]) == step)
            ef = next(r for r in select_stats(implementation, "absE") if int(r["step"]) == step)
            low = f"{float(ne['min']):.6e}" if ne['min'] else "全て非有限"
            high = f"{float(ne['max']):.6e}" if ne['max'] else "全て非有限"
            lines.append(f"| {label} | {step} | {low} | {high} | {ne['negative']} | {int(ef['nan']) + int(ef['inf'])} |")
    lines += ["", "保存済みPython出力ではstep 4800まで全5物理量が有限で、電子密度は非負でした。",
              "step 4900では電子密度に負値が生じ、換算電場の大きさabsEは全20000セルでInfとなっています。",
              "step 5000までプログラムは終了していますが、後半の異常値は有効なシミュレーション結果として扱えません。",
              "条件合わせCでもstep 4900・5000で負の電子密度と全セルInfのabsEが出現しました。",
              "破綻後の巨大な数値は両言語で一致していません。この区間は数値精度の評価から除き、異常値の有無で判定しています。",
              "初めて異常になった正確な更新番号は、100 step間隔の既存保存データからは特定できません。", "",
              "## 留意点", "",
              "元CとPythonには、背景電離とPoisson反復の差があります。元Cの最終結果との不一致だけでは移植ミスとは判断できません。",
              "また、元Cの符号付きMaxPhiでは負電位のときに正しい相対更新量を評価できず、未初期化pphiにも依存します。",
              "元Cが有限値で終了すること自体は、十分に収束した正しい解の証明にはなりません。",
              "", "### Poisson収束判定の独立した確認", "",
              "`poisson_known_solution.c` により、12×16格子、上下端0 V、外側半径勾配0の既知解",
              "`phi=±4*z*(H-z)/H²` を与えて両ソルバを検査しました。初期電位は両方とも明示的にゼロにしています。",
              "許容値は実際の計算と同じ更新量の1e-6です。", "",
              "| ソルバ | 正の既知解: 最大絶対誤差 | 負の既知解: 最大絶対誤差 |",
              "|---|---:|---:|",
              "| 元CのGS・符号付きMaxPhi | 5.283056e-5 | 9.898438e-1 |",
              "| 条件合わせCのSOR・絶対値MaxPhi | 1.795449e-5 | 1.795449e-5 |", "",
              "元Cは負の既知解に対してほとんど未収束の状態で終了しました。この小格子テストは収束判定の問題を確認するもので、",
              "放電計算の各stepで実際にどの程度の誤差を生じたかを直接測定するものではありません。",
              "各実行ログは `original_c/poisson_known_solution.log` と `matched_c/poisson_known_solution.log` に保存しています。", "",
              "この比較は出力一致・数値の有限性を検証するものであり、物理モデルや長時間安定性の妥当性を保証するものではありません。", "",
              "## 保存物と再実行", "",
              "- `comparison_overview.png` / `.pdf`: 密度・電場・差・異常セル数の比較図",
              "- `comparison_by_step.csv`: 各step・各物理量の最大絶対差、スケール誤差、相対L2誤差、完全一致判定",
              "- `field_statistics.csv`: 各実装の最小・最大・負値・NaN/Infの個数",
              "- `summary.json`: 比較結果の要約",
              "- `metadata.json`: ソースSHA-256、ビルドコマンド、実行終了コード、実行条件",
              "- `python_output_sha256.json`: 比較に使用した既存Python CSVのSHA-256",
              "- `original_c/` / `matched_c/`: 検証用ソース、I/Oラッパー、実行ファイル、ビルド・実行ログ",
              "- `original_c/stream_run.log` / `matched_c/stream_run.log`: 最終検証の全step進捗ログ", "",
              "最終検証のC出力全体はディスクに保存せず、パイプで比較した集計表だけを保存しています。",
              "途中のファイル出力方式で生成した分のC出力は、検証用フォルダ内の `.csv.gz` に可逆圧縮して残しています。",
              "そのうち既存のユーザーCSVと改行以外が完全一致する重複分は削除し、SHA-256と参照先の対応を",
              "`interrupted_output_deduplication.json` に保存しました。削除対象は今回生成した検証用コピーだけです。",
              "初回のC実行はディスク空き容量ゼロにより出力途中で停止しました。検証用出力だけを圧縮して容量を確保し、",
              "圧縮付きの再実行も試みましたが、空き容量が安定しないため停止し、最終的にパイプ方式で完走させました。",
              "初回の状態は `attempt_1_disk_full.json` に保存しました。出力容量による停止は数値解の判定には用いていません。",
              "既存のPython CSVと元CのCSVはそのまま保持しています。", "",
              "再実行（streamer_pythonフォルダで実行、出力は検証用フォルダ内に保存）:", "",
              "```powershell", "python tests/validate_c_outputs.py --prepare-only", "python tests/validate_c_stream.py",
              "python tests/report_c_validation.py", "```", "",
              f"元CソースSHA-256: `{metadata['source_sha256']}`", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(ROOT / "REPORT.md")


if __name__ == "__main__":
    main()

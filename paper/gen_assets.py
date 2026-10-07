"""Generates results_macros.tex, table_sweep.tex and table_beyond.tex from
results/metrics.json. Run after long_range_pfn.evaluate:

    .venv/bin/python paper/gen_assets.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(HERE, "..", "results", "metrics.json")


def cell(d, metric="nll", bold=False):
    if d[metric] is None:
        return "--"
    se = d.get(metric + "_se")
    m = f"{d[metric]:.3f}"
    if bold:
        m = f"\\textbf{{{m}}}"
    if se is not None:
        m += f"\\,{{\\scriptsize$\\pm${se:.3f}}}"
    return m


def main():
    with open(METRICS) as f:
        metrics = json.load(f)
    sweep = {int(k): v for k, v in metrics["sweep"].items()}
    beyond = {int(k): v for k, v in metrics["beyond_window"].items()}
    chunks = {int(k): v for k, v in metrics.get("chunks", {}).items()}

    # --- headline macros -------------------------------------------------
    d10 = sweep[10]
    gap = d10["teacher_local"]["nll"] - d10["teacher_full"]["nll"]
    closed = d10["teacher_local"]["nll"] - d10["student"]["nll"]
    close_pct = 100.0 * closed / gap
    trunc_gap_max = max(
        v["teacher_local"]["nll"] - v["teacher_full"]["nll"] for v in sweep.values()
    )
    kl_redux = d10["student_null_z"]["kl"] / d10["student"]["kl"]

    # beyond-window: mean realized gain and headroom past the window
    past = [N for N in beyond if N > 80]
    gain = sum(
        beyond[N]["teacher_window"]["nll"] - beyond[N]["student_stream"]["nll"]
        for N in past
    ) / len(past)
    headroom = sum(
        beyond[N]["gp_exact_window"]["nll"] - beyond[N]["gp_exact_full"]["nll"]
        for N in past
    ) / len(past)

    # certificate margin: how far below the D-only exact-GP bound the student
    # falls, min and max across the |D| sweep (student NLL - gp_exact_local NLL,
    # negative means below the bound; report the magnitude)
    margins = [
        sweep[n]["gp_exact_local"]["nll"] - sweep[n]["student"]["nll"] for n in sweep
    ]
    margin_min, margin_max = min(margins), max(margins)

    # recursion cost: KL(teacher||student) at T=1 (single-shot) vs T=5 chunks,
    # both in absolute terms and relative to the single-shot ceiling
    chunk_min = chunks[min(chunks)]["kl"] if chunks else None
    chunk_max = chunks[max(chunks)]["kl"] if chunks else None

    with open(os.path.join(HERE, "results_macros.tex"), "w") as f:
        f.write(f"\\newcommand{{\\shortclosepct}}{{{close_pct:.0f}\\%}}\n")
        f.write(f"\\newcommand{{\\truncgapmax}}{{{trunc_gap_max:.2f}}}\n")
        f.write(f"\\newcommand{{\\klreduxfactor}}{{{kl_redux:.1f}$\\times$}}\n")
        f.write(f"\\newcommand{{\\beyondgain}}{{{gain:.3f}}}\n")
        f.write(f"\\newcommand{{\\beyondheadroom}}{{{headroom:.3f}}}\n")
        f.write(
            f"\\newcommand{{\\beyondgainpct}}{{{100.0 * gain / headroom:.0f}\\%}}\n"
        )
        f.write(f"\\newcommand{{\\certmarginmin}}{{{margin_min:.2f}}}\n")
        f.write(f"\\newcommand{{\\certmarginmax}}{{{margin_max:.2f}}}\n")
        if chunk_min is not None:
            f.write(f"\\newcommand{{\\chunkonekl}}{{{chunk_min:.3f}}}\n")
            f.write(f"\\newcommand{{\\chunkfivekl}}{{{chunk_max:.3f}}}\n")

    # --- sweep table ------------------------------------------------------
    lines = [
        "\\begin{tabular}{r cccccc ccc}",
        "\\toprule",
        " & \\multicolumn{6}{c}{NLL (nats)} & \\multicolumn{3}{c}{KL to teacher-full (nats)}\\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-10}",
        "$|D|$ & GP exact & teacher & teacher & LR-PFN & LR-PFN & no-$z$ & teacher & LR-PFN & no-$z$\\\\",
        " & (full) & full & local & & (3 chunks) & (abl.) & local & & (abl.)\\\\",
        "\\midrule",
    ]
    for n in sorted(sweep):
        v = sweep[n]
        lines.append(
            f"{n} & {cell(v['gp_exact_full'])} & {cell(v['teacher_full'])}"
            f" & {cell(v['teacher_local'])} & {cell(v['student'], bold=True)}"
            f" & {cell(v['student_stream'])} & {cell(v['student_null_z'])}"
            f" & {cell(v['teacher_local'], 'kl')} & {cell(v['student'], 'kl', bold=True)}"
            f" & {cell(v['student_null_z'], 'kl')}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(HERE, "table_sweep.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- beyond-window table ---------------------------------------------
    lines = [
        "\\begin{tabular}{r cccc}",
        "\\toprule",
        "$|D_{\\mathrm{tot}}|$ & GP exact (full) & GP exact (window) & teacher (window) & LR-PFN (stream)\\\\",
        "\\midrule",
    ]
    for n in sorted(beyond):
        v = beyond[n]
        lines.append(
            f"{n} & {cell(v['gp_exact_full'])} & {cell(v['gp_exact_window'])}"
            f" & {cell(v['teacher_window'])} & {cell(v['student_stream'], bold=True)}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(HERE, "table_beyond.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- defensive baselines table (appendix, optional) -----------------
    bpath = os.path.join(HERE, "..", "results", "baselines.json")
    if os.path.exists(bpath):
        write_baselines_table(bpath)

    print("wrote results_macros.tex, table_sweep.tex, table_beyond.tex")
    print(f"gap closed at |D|=10: {close_pct:.1f}% | KL reduction: {kl_redux:.1f}x")
    print(f"beyond-window gain: {gain:.3f} nats of {headroom:.3f} headroom")


def write_baselines_table(bpath):
    with open(bpath) as f:
        bl = {int(k): v for k, v in json.load(f).items()}
    totals = sorted(bl)
    # (key, label, reports NLL, is a full-context baseline)
    rows = [
        ("teacher_window", "Teacher (80-pt window)", True, False),
        ("student", "\\textbf{LongRange-PFN (bounded)}", True, False),
        ("gp_exact_full", "GP exact (full, oracle)", True, True),
        ("tabpfn_full", "TabPFN (full context)", True, True),
        ("gp_rbf_full", "GP-RBF fitted (full)", True, True),
        ("krr", "Kernel ridge (full)", False, True),
        ("knn", "$k$-NN (full)", False, True),
        ("rf", "Random forest (full)", False, True),
    ]

    def pair(v):
        m, se = v
        return f"{m:.3f}\\,{{\\scriptsize$\\pm${se:.3f}}}" if m is not None else "--"

    header = "Method & " + " & ".join(f"$N={n}$" for n in totals)
    ncol = len(totals) + 1

    def block(metric):
        out = []
        for key, label, has_nll, _ in rows:
            if metric == "nll" and not has_nll:
                continue
            cells = [pair(bl[n][key][metric]) for n in totals]
            out.append(f"{label} & " + " & ".join(cells) + "\\\\")
        return out

    lines = [
        "\\begin{tabular}{l" + "c" * len(totals) + "}",
        "\\toprule",
        "\\multicolumn{" + str(ncol) + "}{c}{\\emph{NLL (nats, lower is better)}}\\\\",
        "\\midrule",
        header + "\\\\",
        "\\midrule",
        *block("nll"),
        "\\midrule",
        "\\multicolumn{" + str(ncol) + "}{c}{\\emph{RMSE (lower is better)}}\\\\",
        "\\midrule",
        header + "\\\\",
        "\\midrule",
        *block("rmse"),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(os.path.join(HERE, "table_baselines.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    n_tab = bl[totals[0]]["tabpfn_full"]["n"]
    print(f"wrote table_baselines.tex (TabPFN N={n_tab})")


if __name__ == "__main__":
    main()

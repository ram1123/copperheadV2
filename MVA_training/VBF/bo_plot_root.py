# bo_plot_root.py
import argparse, csv, json, math, os, sys
from collections import defaultdict, OrderedDict

import ROOT

# ---------------------------
# IO helpers
# ---------------------------


def _mkdir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def _read_jsonl(jsonl_path):
    """
    Read trials from a JSON Lines file (one JSON object per line).
    Keeps only status=='ok' and numeric AUC.
    Expects keys: trial_index, auc, duration_sec (optional), params (dict).
    """
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            status = str(obj.get("status", "")).lower()
            if status != "ok":
                continue

            try:
                auc = float(obj["auc"])
            except Exception:
                continue

            try:
                trial_idx = int(obj.get("trial_index", len(rows) + 1))
            except Exception:
                trial_idx = len(rows) + 1

            dur = None
            try:
                dur = float(obj.get("duration_sec", ""))
            except Exception:
                pass

            params = obj.get("params", {}) or {}
            if not isinstance(params, dict):
                params = {}

            rows.append({"trial": trial_idx, "auc": auc, "dur": dur, "params": params})

    rows.sort(key=lambda x: x["trial"])
    return rows


def _read_csv(csv_path):
    """
    Backward-compat: read old CSV (with params_json column).
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r.get("status", "")).lower() != "ok":
                continue
            try:
                auc = float(r["auc"])
            except Exception:
                continue

            # params may be in a 'params' column (JSON) or 'params_json' string
            params = {}
            raw = r.get("params")
            if raw:
                try:
                    params = json.loads(raw)
                except Exception:
                    params = {}
            else:
                raw = r.get("params_json", "")
                if raw:
                    try:
                        params = json.loads(raw)
                    except Exception:
                        params = {}

            try:
                trial_idx = int(r.get("trial_index", len(rows) + 1))
            except Exception:
                trial_idx = len(rows) + 1

            dur = None
            try:
                dur = float(r.get("duration_sec", ""))
            except Exception:
                pass

            rows.append({"trial": trial_idx, "auc": auc, "dur": dur, "params": params})

    rows.sort(key=lambda x: x["trial"])
    return rows


def _read_trials(path):
    """
    Auto-detect by extension. Supports:
      - .jsonl : JSON Lines (recommended)
      - .csv   : legacy CSV
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl" or ext == ".json":
        # accept .json if it's jsonl-style (one object per line)
        return _read_jsonl(path)
    elif ext == ".csv":
        return _read_csv(path)
    else:
        # try JSONL first, then CSV
        try:
            return _read_jsonl(path)
        except Exception:
            return _read_csv(path)


# ---------------------------
# Plotting helpers (ROOT)
# ---------------------------


def _new_canvas(name="c", w=900, h=700):
    c = ROOT.TCanvas(name, name, w, h)
    c.SetGrid(1, 1)
    return c


def _draw_graph(x, y, xtitle, ytitle, title, outpath, xlog=False):
    n = len(x)
    g = ROOT.TGraph(n)
    for i in range(n):
        g.SetPoint(i, float(x[i]), float(y[i]))
    c = _new_canvas()
    if xlog:
        c.SetLogx()
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetTitle(f"{title};{xtitle};{ytitle}")
    g.Draw("AP")
    c.SaveAs(outpath)
    c.Close()


def _encode_categorical(vals):
    uniq = []
    seen = set()
    for v in vals:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    mapping = OrderedDict((v, i + 1) for i, v in enumerate(uniq))  # start at 1
    enc = [mapping[v] for v in vals]
    return enc, mapping


def _legend_for_mapping(mapping, title):
    leg = ROOT.TLegend(0.15, 0.75, 0.55, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader(title, "C")
    for k, idx in mapping.items():
        leg.AddEntry(0, f"{idx}: {k}", "")
    return leg


def _draw_categorical(xvals, auc, mapping, xtitle, ytitle, title, outpath):
    n = len(xvals)
    g = ROOT.TGraph(n)
    for i in range(n):
        g.SetPoint(i, float(xvals[i]), float(auc[i]))
    c = _new_canvas()
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetTitle(f"{title};{xtitle};{ytitle}")
    g.Draw("AP")
    leg = _legend_for_mapping(mapping, f"{xtitle} mapping")
    leg.Draw()
    c.SaveAs(outpath)
    c.Close()


def _write_top_tables(rows, outdir, topN=10):
    rows_sorted = sorted(rows, key=lambda r: r["auc"], reverse=True)
    top = rows_sorted[:topN]
    # CSV
    csv_path = os.path.join(outdir, "top_trials.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "trial", "auc", "duration_sec", "params_json"])
        for i, r in enumerate(top, 1):
            w.writerow(
                [
                    i,
                    r["trial"],
                    f"{r['auc']:.6f}",
                    f"{r['dur']:.3f}" if r["dur"] else "",
                    json.dumps(r["params"]),
                ]
            )
    # Markdown
    md_path = os.path.join(outdir, "top_trials.md")
    with open(md_path, "w") as f:
        f.write("| rank | trial | AUC | duration [s] | params_json |\n")
        f.write("|:----:|:-----:|:---:|:------------:|:------------|\n")
        for i, r in enumerate(top, 1):
            dur = f"{r['dur']:.3f}" if r["dur"] else ""
            f.write(
                f"| {i} | {r['trial']} | {r['auc']:.6f} | {dur} | `{json.dumps(r['params'])}` |\n"
            )


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "trials", help="Path to trials file (.jsonl preferred, .csv supported)"
    )
    ap.add_argument("outdir", help="Output directory for plots/tables")
    ap.add_argument("--topN", type=int, default=10)
    args = ap.parse_args()

    _mkdir(args.outdir)
    rows = _read_trials(args.trials)
    if not rows:
        print("No valid trials found.")
        sys.exit(1)

    # --- Optimization trace (AUC vs trial) ---
    _draw_graph(
        [r["trial"] for r in rows],
        [r["auc"] for r in rows],
        "Iteration",
        "AUC",
        "Optimization Trace",
        os.path.join(args.outdir, "auc_trace.pdf"),
        xlog=False,
    )

    # Collect per-param series
    auc = [r["auc"] for r in rows]
    P = defaultdict(list)
    for r in rows:
        p = r["params"]
        # numerics
        P["hidden0"].append(float(p.get("hidden0", "nan")))
        P["dropout"].append(float(p.get("dropout", "nan")))
        P["batch_size"].append(float(p.get("batch_size", "nan")))
        P["lr"].append(float(p.get("lr", "nan")))
        P["weight_decay"].append(float(p.get("weight_decay", "nan")))
        # categoricals
        P["activation"].append(str(p.get("activation", "")))
        P["optimizer"].append(str(p.get("optimizer", "")))
        P["loss_name"].append(str(p.get("loss_name", "")))

    # Numeric plots
    _draw_graph(
        P["hidden0"],
        auc,
        "hidden0",
        "AUC",
        "AUC vs hidden0",
        os.path.join(args.outdir, "auc_vs_hidden0.pdf"),
    )
    _draw_graph(
        P["dropout"],
        auc,
        "dropout",
        "AUC",
        "AUC vs dropout",
        os.path.join(args.outdir, "auc_vs_dropout.pdf"),
    )
    _draw_graph(
        P["batch_size"],
        auc,
        "batch_size",
        "AUC",
        "AUC vs batch_size",
        os.path.join(args.outdir, "auc_vs_batch_size.pdf"),
    )

    # Log10 axes for lr, weight_decay
    def _safe_log10(xs):
        out = []
        for v in xs:
            try:
                v = float(v)
                out.append(math.log10(v) if v > 0 else float("nan"))
            except Exception:
                out.append(float("nan"))
        return out

    _draw_graph(
        _safe_log10(P["lr"]),
        auc,
        "log10(lr)",
        "AUC",
        "AUC vs learning rate",
        os.path.join(args.outdir, "auc_vs_lr.pdf"),
    )
    _draw_graph(
        _safe_log10(P["weight_decay"]),
        auc,
        "log10(weight_decay)",
        "AUC",
        "AUC vs weight decay",
        os.path.join(args.outdir, "auc_vs_weight_decay.pdf"),
    )

    # Categorical plots
    for cat in ["activation", "optimizer", "loss_name"]:
        enc, mapping = _encode_categorical(P[cat])
        _draw_categorical(
            enc,
            auc,
            mapping,
            cat,
            "AUC",
            f"AUC vs {cat}",
            os.path.join(args.outdir, f"auc_vs_{cat}.pdf"),
        )

    # Top-N tables
    _write_top_tables(rows, args.outdir, topN=args.topN)


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    main()

#!/usr/bin/env python3
"""
Build the generated parts of the flag-skim 2024 v2 report deck (FlagSkim_2024_v2.tex): booktabs tables read
straight from the result CSVs (gen_tables.tex) and the control-plot slides (gen_plots_main.tex,
gen_plots_backup.tex), copying the plots into images/ so the deck builds without /work paths.

How to run (from the worktree root, after scripts/flag_skim_control_plots.py and the yield/cutflow checks in
docs/flag_skim.md § "Checks: commands"):
  /work/pixi/global/.pixi/envs/default/bin/python reports/FlagSkim_2024_v2/make_report_inputs.py
Example (full rebuild):
  /work/pixi/global/.pixi/envs/default/bin/python reports/FlagSkim_2024_v2/make_report_inputs.py && \
  cd reports/FlagSkim_2024_v2 && PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/latest/bin/x86_64-linux:$PATH \
  latexmk -pdf -interaction=nonstopmode FlagSkim_2024_v2.tex
"""

import glob
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V = "validation/flag_skim/2024_v2"
CP = f"{V}/control_plots"
NT = "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean"
LABEL = "Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit"
OCT26 = "Run3_nanoAODv15_FilterEvents_Oct26_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit"
VALID = ["looseID", "looseIDIP", "looseIDlepIPVeto", "official", "ipOnly", "lepIPVeto",
         "tightID", "tightIDIP", "tightIDlepIPVeto"]
DESC = {"looseID": ("Loose", "--", "2$\\mu$+$e$-veto"), "looseIDIP": ("Loose", "\\checkmark", "2$\\mu$+$e$-veto"),
        "looseIDlepIPVeto": ("Loose", "\\checkmark", "lepton veto"), "official": ("Medium", "--", "2$\\mu$+$e$-veto"),
        "ipOnly": ("Medium", "\\checkmark", "2$\\mu$+$e$-veto"), "lepIPVeto": ("Medium", "\\checkmark", "lepton veto"),
        "tightID": ("Tight", "--", "2$\\mu$+$e$-veto"), "tightIDIP": ("Tight", "\\checkmark", "2$\\mu$+$e$-veto"),
        "tightIDlepIPVeto": ("Tight", "\\checkmark", "lepton veto"),
        "trigDR0p1": ("Medium", "--", "2$\\mu$+$e$-veto, $\\Delta R<0.1$"),
        "nMu2NoLepVeto": ("Medium", "--", "2$\\mu$, \\textbf{no} $e$-veto (diag.)")}
REGIONS = ["jj_both_central", "jj_one_fwd25_one_central", "jj_both_fwd25",
           "jj_combined_both_central_one_fwd25_one_central_both_fwd25"]


def tt(s):
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def table(cols, rows, fmt="l"):
    spec = fmt + "r" * (len(cols) - 1) if len(fmt) == 1 else fmt
    out = ["\\resizebox{\\textwidth}{!}{\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(cols) + " \\\\",
           "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(out)


def f(x, nd=1):
    return f"{x:,.{nd}f}"


def gen_tables():
    T = {}
    # ---- inclusive yields: validation vs Oct26/Oct23 and per selection
    y = pd.read_csv(f"{V}/yields_per_group.csv")
    g = y.set_index(["group", "selection"])
    groups = ["Data", "DY", "Top", "VV", "EWK", "ggH", "VBF"]
    rows = []
    for grp in groups:
        r = [grp]
        for a, b in (("ref_Oct26", "flag_official"), ("ref_Oct23", "flag_lepIPVeto")):
            ra, rb = g.loc[(grp, a)], g.loc[(grp, b)]
            r += [f"{int(ra.n):,}", f"{int(rb.n):,}", f"{ra.w:,.2f}", f"{rb.w:,.2f}",
                  "\\textcolor{green!50!black}{=}" if (ra.n == rb.n and abs(ra.w - rb.w) <= 1e-6 * abs(ra.w)) else "\\textcolor{red}{$\\neq$}"]
        rows.append(r)
    T["incl_validation"] = table(["group", "Oct26 $n$", "official $n$", "Oct26 $w$", "official $w$", "",
                                  "Oct23 $n$", "lepIPVeto $n$", "Oct23 $w$", "lepIPVeto $w$", ""], rows)
    rows = []
    for sel in VALID + ["trigDR0p1", "nMu2NoLepVeto"]:
        k = f"flag_{sel}"
        idd, ip, vet = DESC[sel]
        d = g.loc[("Data", k)]
        sig = g.loc[("ggH", k)].w_win + g.loc[("VBF", k)].w_win
        bkg = sum(g.loc[(b, k)].w_win for b in ["DY", "Top", "VV", "EWK"])
        bkg_all = sum(g.loc[(b, k)].w for b in ["DY", "Top", "VV", "EWK"])
        rows.append([tt(sel), idd, ip, vet, f"{int(d.n):,}", f"{bkg_all:,.0f}", f"{d.n / bkg_all:.3f}",
                     f"{sig:.1f}", f"{bkg:,.0f}"])
    T["incl_selections"] = table(["selection", "ID", "IP", "leptons", "data (all $m$)", "MC bkg (all $m$)", "data/MC",
                                  "S (110--150)", "B (110--150)"], rows, "llcl" + "r" * 5)
    # ---- VBF h-sideband yields (get_yields)
    a = pd.read_csv(f"{V}/get_yields/yield_summary_all_cases_2024.csv").set_index("case")
    rows = []
    for sel in VALID + ["trigDR0p1", "nMu2NoLepVeto"]:
        r = a.loc[sel]
        idd, ip, vet = DESC[sel]
        rows.append([tt(sel), idd, ip, vet, f"{int(r.data):,}", f"{r.ggH:.3f}", f"{r.VBF:.3f}", f"{r.bkg:,.0f}",
                     f"{r.DY:,.0f}", f"{r.TT:,.0f}", f"{r.data_over_mc:.3f}", f"{1000 * r.s_over_sqrt_sb:.2f}"])
    T["vbf_hsb_yields"] = table(["selection", "ID", "IP", "leptons", "data", "ggH", "VBF", "bkg", "DY", "TT", "data/MC",
                                 "$10^{3}\\,S/\\sqrt{S+B}$"], rows, "llcl" + "r" * 8)
    # ---- cutflows (looseIDIP, dR<0.4)
    for reg in ("h-sidebands", "z-peak", "signal"):
        w = pd.read_csv(f"{V}/cutflow_looseIDIP_{reg}/cutflow_wide.csv")
        w = w[w.dr == "0p4"].sort_values("step_idx")
        rows = []
        for _, r in w.iterrows():
            data = "--" if pd.isna(r.data) else f"{int(r.data):,}"
            dmc = "--" if pd.isna(r["data/bkg"]) else f"{r['data/bkg']:.3f}"
            rows.append([r.step.replace("_", "\\_").replace(">", "$>$").replace("<", "$<$"), data, f"{r.GGH:.3g}",
                         f"{r.VBF:.3g}", f"{r.DY:,.4g}", f"{r.TT:,.4g}", f"{r.EWK:,.4g}", f"{r.VV:,.4g}",
                         f"{r.background:,.5g}", dmc, f"{r['S/sqrt(S+B)']:.4f}"])
        T[f"cutflow_{reg}"] = table(["step", "data", "ggH", "VBF", "DY", "TT", "EWK", "VV", "bkg", "data/bkg",
                                     "$S/\\sqrt{S+B}$"], rows)
    # ---- dR scan
    rows = []
    sb = pd.read_csv(f"{V}/cutflow_looseIDIP_h-sidebands/cutflow_wide.csv")
    zp = pd.read_csv(f"{V}/cutflow_looseIDIP_z-peak/cutflow_wide.csv")
    sr = pd.read_csv(f"{V}/cutflow_looseIDIP_signal/cutflow_wide.csv")

    def at(df, dr, idx, col):
        return df[(df.dr == dr) & (df.step_idx == idx)][col].iloc[0]
    ref = at(sb, "0p4", 5, "data")
    for dr in ["0p4", "0p3", "0p2", "0p15", "0p1"]:
        d5 = at(sb, dr, 5, "data")
        rows.append([dr.replace("p", "."), f"{int(d5):,}", f"{100 * (ref - d5) / ref:.4f}\\%",
                     f"{int(at(zp, dr, 10, 'data')):,}", f"{at(zp, dr, 10, 'background'):,.1f}",
                     f"{int(at(sb, dr, 10, 'data')):,}", f"{at(sb, dr, 10, 'background'):,.1f}",
                     f"{at(sr, dr, 10, 'signal'):.4f}", f"{at(sr, dr, 10, 'background'):,.1f}",
                     f"{at(sr, dr, 11, 'signal'):.3f}", f"{at(sr, dr, 11, 'background'):,.1f}"])
    T["dr_scan"] = table(["$\\Delta R_{\\rm trig}<$", "data after trig.", "lost vs 0.4", "Z-peak VBF data",
                          "Z-peak VBF bkg", "SB VBF data", "SB VBF bkg", "SR VBF $S$", "SR VBF $B$",
                          "115--135 $S$", "115--135 $B$"], rows)
    # ---- significance / limits from stage-3
    rows = []

    def row_2024(path):
        # summaries of multi-year runs have one row per year: take 2024's, never the last row
        d = pd.read_csv(path)
        d = d[d.year.astype(str) == "2024"]
        if len(d) != 1:
            raise ValueError(f"expected one 2024 row in {path}, got {len(d)}")
        return d.iloc[0]

    def sig_row(name, pattern):
        r = [name]
        for reg in REGIONS:
            fs = glob.glob(pattern.format(reg=reg) + "/score_*/vbf_significance_summary_*.csv")
            if not fs:
                r.append("--")
                continue
            s = row_2024(fs[0])
            r.append(f"{s.significance:.3f} ({s.significance_statonly:.3f})")
        fl = glob.glob(pattern.format(reg=REGIONS[-1]) + "/score_*/vbf_expected_limit_summary_*.csv")
        r.append(f"{row_2024(fl[0]).expected_limit_median:.2f}" if fl else "--")
        return r
    rows.append(sig_row("Oct26 (reference)", f"{NT}/{OCT26}/stage3_datacards_Oct26_2026_Syst_DNNRay050_{{reg}}"))
    for sel in ["official", "lepIPVeto", "ipOnly", "looseID", "tightID", "tightIDlepIPVeto", "trigDR0p1",
                "nMu2NoLepVeto"]:
        rows.append(sig_row(tt(sel), f"{NT}/{LABEL}/stage3_datacards_Oct30_2026_Syst_DNNRay050_FSv2_{sel}_{{reg}}"))
    T["significance"] = table(["case", "both central", "one fwd/one central", "both fwd", "\\textbf{combined}",
                               "exp. limit (comb.)"], rows)
    with open(f"{HERE}/gen_tables.tex", "w") as fh:
        for k, v in T.items():
            fh.write(f"\\newcommand{{\\tab{k.replace('_', '').replace('-', '')}}}{{%\n{v}\n}}\n")
    print("wrote gen_tables.tex:", ", ".join(T))


# ---------------------------------------------------------------- plot slides
GROUPS = {
    "Dimuon kinematics": ["dimuon_mass", "dimuon_pt", "dimuon_pt_log", "dimuon_rapidity", "dimuon_eta", "dimuon_phi",
                          "dimuon_cos_theta_cs", "dimuon_phi_cs", "dimuon_ebe_mass_res", "dimuon_ebe_mass_res_rel"],
    "Muon kinematics": ["mu1_pt", "mu2_pt", "mu1_eta", "mu2_eta", "mu1_phi", "mu2_phi", "mu1_pt_over_mass",
                        "mu2_pt_over_mass"],
    "Leading and subleading jets": ["jet1_pt", "jet2_pt", "jet1_eta", "jet2_eta", "jet1_phi", "jet2_phi"],
    "Dijet": ["jj_mass", "jj_mass_log", "jj_dEta", "jj_dPhi", "jj_pt", "njets"],
    "Other MVA inputs": ["nsoftjets5", "htsoft2", "rpt", "ll_zstar_log", "mmj_min_dEta", "mmj_min_dPhi",
                         "pt_centrality", "zeppenfeld", "nBtagLoose", "PV_npvsGood"],
}
VBF_DNN = ["dimuon_mass", "dimuon_pt", "dimuon_pt_log", "dimuon_rapidity", "dimuon_ebe_mass_res",
           "dimuon_ebe_mass_res_rel", "dimuon_cos_theta_cs", "dimuon_phi_cs", "jet1_pt", "jet1_eta", "jet1_phi",
           "jet2_pt", "jet2_eta", "jet2_phi", "jj_mass", "jj_mass_log", "jj_dEta", "nsoftjets5", "htsoft2", "rpt",
           "ll_zstar_log", "mmj_min_dEta", "pt_centrality"]
GGH_BDT = ["dimuon_cos_theta_cs", "dimuon_phi_cs", "dimuon_rapidity", "dimuon_pt", "jet1_eta", "jet2_eta", "jet1_pt",
           "jet2_pt", "jj_dEta", "jj_dPhi", "jj_mass", "mmj_min_dEta", "mmj_min_dPhi", "mu1_eta", "mu1_pt_over_mass",
           "mu2_eta", "mu2_pt_over_mass", "zeppenfeld", "njets"]
PS_TITLE = {"zpeak_incl": "Z peak (70--110 GeV), inclusive", "zpeak_ggh": "Z peak, ggH category",
            "zpeak_vbf": "Z peak, VBF category", "hsb_ggh": "H sidebands, ggH category",
            "hsb_vbf": "H sidebands, VBF category"}


def img(kind, *parts):
    src = os.path.join(CP, kind, *parts) + ".pdf"
    rel = os.path.join("images", kind, *parts) + ".pdf"
    dst = os.path.join(HERE, rel)
    if not os.path.exists(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return rel


def grid_frames(title, paths, per=6, note=""):
    out = []
    for i in range(0, len(paths), per):
        chunk = [p for p in paths[i:i + per] if p]
        if not chunk:
            continue
        n = (i // per) + 1
        tot = (len(paths) + per - 1) // per
        out.append(f"\\begin{{frame}}{{{title}" + (f" ({n}/{tot})" if tot > 1 else "") + "}")
        if note:
            out.append(f"\\vspace{{-0.2cm}}{{\\scriptsize {note}}}\\\\")
        out.append("\\centering")
        for j, p in enumerate(chunk):
            if per == 3:  # one row of large plots (main body)
                out.append(f"\\includegraphics[width=0.325\\textwidth]{{{p}}}%")
                continue
            # height-based: two rows of square plots must fit the 16:9 frame
            out.append(f"\\includegraphics[height=0.37\\textheight]{{{p}}}%")
            if (j + 1) % (3 if per == 6 else 4) == 0:
                out.append("\\\\")
        out.append("\\end{frame}")
    return out


def stack_pairs(title, ps, variables):
    """One variable per frame: official (left) vs looseIDIP (right) data/MC, two large plots."""
    out = []
    for v in variables:
        out.append(f"\\begin{{frame}}{{{title}: {v.replace('_', chr(92) + '_')}}}")
        out.append("\\begin{columns}[T]")
        for sel in ("official", "looseIDIP"):
            p = img("stack", sel, ps, v)
            out.append("\\begin{column}{0.49\\textwidth}\\centering\\footnotesize\\textbf{" + sel + "}\\\\")
            if p:
                out.append(f"\\includegraphics[height=0.78\\textheight]{{{p}}}")
            out.append("\\end{column}")
        out.append("\\end{columns}")
        out.append("\\end{frame}")
    return out


def gen_plots():
    main, backup = [], []
    note_cmp = ("Top: data/MC per selection (blue Loose, black Medium, red Tight; dashed no IP, solid +IP, "
                "dotted +lepton veto). Bottom: data shape / official data shape.")
    # Z peak inclusive: muons, dimuon, jets, dijet
    main.append("\\section{Control plots: Z peak}")
    for grp in ["Dimuon kinematics", "Muon kinematics", "Leading and subleading jets", "Dijet"]:
        main += grid_frames(f"{grp}: {PS_TITLE['zpeak_incl']}", [img("compare", "zpeak_incl", v) for v in GROUPS[grp]],
                            per=3, note=note_cmp)
    main += stack_pairs(f"Data/MC: {PS_TITLE['zpeak_incl']}", "zpeak_incl",
                        ["dimuon_pt", "mu1_pt", "jet1_pt", "jj_mass"])
    main.append("\\section{Control plots: VBF DNN inputs}")
    main += grid_frames(f"VBF DNN inputs: {PS_TITLE['hsb_vbf']}", [img("compare", "hsb_vbf", v) for v in VBF_DNN],
                        per=3, note=note_cmp)
    main += stack_pairs(f"Data/MC: {PS_TITLE['hsb_vbf']}", "hsb_vbf", ["jj_mass", "jj_dEta", "jet1_pt", "jet2_eta"])
    main.append("\\section{Control plots: ggH BDT inputs}")
    main += grid_frames(f"ggH BDT inputs: {PS_TITLE['hsb_ggh']}", [img("compare", "hsb_ggh", v) for v in GGH_BDT],
                        per=3, note=note_cmp)
    main += stack_pairs(f"Data/MC: {PS_TITLE['hsb_ggh']}", "hsb_ggh", ["dimuon_pt", "njets", "jet1_pt", "jj_mass"])
    # backup: every variable, every phase space, comparison; stacks for both selections in all phase spaces
    allv = [v for vs in GROUPS.values() for v in vs]
    for ps in ["zpeak_incl", "zpeak_ggh", "zpeak_vbf", "hsb_ggh", "hsb_vbf"]:
        backup += grid_frames(f"Backup -- all variables: {PS_TITLE[ps]}", [img("compare", ps, v) for v in allv],
                              note=note_cmp)
    for sel in ("official", "looseIDIP"):
        for ps in ["zpeak_incl", "hsb_vbf", "hsb_ggh"]:
            backup += grid_frames(f"Backup -- data/MC {sel}: {PS_TITLE[ps]}", [img("stack", sel, ps, v) for v in allv],
                                  per=8)
    for name, lines in (("gen_plots_main.tex", main), ("gen_plots_backup.tex", backup)):
        with open(f"{HERE}/{name}", "w") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote gen_plots_main.tex ({sum(l.startswith(chr(92) + 'begin{frame}') for l in main)} frames), "
          f"gen_plots_backup.tex ({sum(l.startswith(chr(92) + 'begin{frame}') for l in backup)} frames)")


if __name__ == "__main__":
    gen_tables()
    gen_plots()

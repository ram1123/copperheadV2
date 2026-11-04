"""Flag-based stage-1 skim (switch do_flag_skim).

One loose preselection (loose muon ID, no muon IP cut, no lepton veto, trigger match at the loosest
dR) replaces the per-scenario ntuples; the tighter muon-ID / IP / lepton-veto / trigger-match-dR
choices are stored as per-event flags, plus the per-WP muon pair as positions in a stored muon list.
An event whose WPs pick different leading pairs gets one output row per distinct pair (pair_rows), each
built from its own pair, so every selection sees the columns a dedicated run would give.
Column reference and the official-selection recipes: docs/flag_skim.md.
"""
import awkward as ak
import numpy as np

MUON_ID_FIELDS = {"looseID": "looseId", "mediumID": "mediumId", "tightID": "tightId"}
TRIGMATCH_DRS = (0.4, 0.3, 0.2, 0.15, 0.1)
N_MUON_LIST = 4

# stored muon-list fields: output name -> (Muon field, fill value for padded slots)
MUON_LIST_FIELDS = {
    "pt": ("pt", -999.0),
    "ptRoch": ("pt_roch", -999.0),
    "eta": ("eta", -999.0),
    "phi": ("phi", -999.0),
    "charge": ("charge", 0),
    "iso": ("pfRelIso04_all", -999.0),
    "dxy": ("dxy", -999.0),
    "dz": ("dz", -999.0),
    "looseId": ("looseId", False),
    "mediumId": ("mediumId", False),
    "tightId": ("tightId", False),
    "passIP": ("pass_ip", False),
    "trigMinDR": ("trig_minDR", 999.0),
    "nanoIdx": ("nano_idx", -1),
}


def dr_tag(dr):
    return "dR" + f"{dr:g}".replace(".", "p")


def wp_tag(id_tag, with_ip):
    return f"{id_tag}_{'IP' if with_ip else 'noIP'}"


def wp_muon_masks(muons, base_sel, muon_ip):
    """Per-WP muon masks; every WP is a subset of base_sel (loose ID, no IP)."""
    masks = {}
    for id_tag, field in MUON_ID_FIELDS.items():
        id_sel = base_sel & muons[field]
        masks[wp_tag(id_tag, False)] = id_sel
        masks[wp_tag(id_tag, True)] = id_sel & muon_ip
    return masks


def leading_pair(muons, sel):
    """Two highest-pt muons passing sel (None when missing); same pad/sort as the processor's mu1/mu2."""
    padded = ak.pad_none(muons[sel], 2)
    padded = padded[ak.argsort(padded.pt, ascending=False)]
    return padded[:, 0], padded[:, 1]


def muon_trig_min_dr(muons, trigger_cands):
    """Per-muon min dR to the trigger candidates (999 when there are none)."""
    pairs = ak.cartesian([muons, trigger_cands], nested=True)
    return ak.fill_none(ak.min(pairs["0"].delta_r(pairs["1"]), axis=2), 999.0)


def trigger_match_flags(m1, m2, lead_pt, drs):
    """Production trigger-match rule (either leading muon matched with pt_roch >= lead_pt) per dR."""
    def matched(m, dr):
        return ak.fill_none((m.trig_minDR <= dr) & (m.pt_roch >= lead_pt), False)
    return {dr: matched(m1, dr) | matched(m2, dr) for dr in drs}


def list_position(sorted_muons, nano_idx):
    """Position of the muon with NanoAOD index nano_idx in sorted_muons, -1 if absent."""
    match = ak.values_astype(sorted_muons.nano_idx == nano_idx, "int8")
    pos = ak.where(ak.sum(match, axis=1) > 0, ak.argmax(match, axis=1), -1)
    return ak.fill_none(pos, -1)


def muon_list_block(sorted_base):
    """Leading N_MUON_LIST base muons (pt-sorted, after FSR) as flat mulist<i>_<var> columns."""
    padded = ak.pad_none(sorted_base, N_MUON_LIST, clip=True)
    out = {"nMuons_base": ak.num(sorted_base, axis=1)}
    for i in range(N_MUON_LIST):
        mu = padded[:, i]
        for name, (field, fill) in MUON_LIST_FIELDS.items():
            col = ak.fill_none(mu[field], fill)
            if isinstance(fill, float):
                col = ak.values_astype(col, np.float32)  # float32 halves the list's size; not used for analysis kinematics
            out[f"mulist{i}_{name}"] = col
    return out


def pair_block(tag, m1, m2, n_wp, sorted_base, prim_idx1, prim_idx2, trig_flags):
    """Columns for one WP pair; prim_idx* = NanoAOD indices of the primary pair (mu1/mu2 columns)."""
    idx1 = ak.fill_none(m1.nano_idx, -1)
    idx2 = ak.fill_none(m2.nano_idx, -1)
    out = {
        f"nMuons_{tag}": n_wp,
        f"pair_{tag}_pos1": list_position(sorted_base, idx1),
        f"pair_{tag}_pos2": list_position(sorted_base, idx2),
        f"pair_{tag}_OS": ak.fill_none(m1.charge * m2.charge == -1, False),
        f"pair_{tag}_mass": ak.fill_none((m1 + m2).mass, -1.0),
        f"pair_{tag}_isPrimary": (idx2 >= 0) & (idx1 == prim_idx1) & (idx2 == prim_idx2),
    }
    for dr, flag in trig_flags.items():
        out[f"pair_{tag}_trigMatch_{dr_tag(dr)}"] = flag
    return out


def _pair_ids(muons, sel):
    m1, m2 = leading_pair(muons, sel)
    return (ak.to_numpy(ak.fill_none(m1.nano_idx, -1)).astype(np.int64),
            ak.to_numpy(ak.fill_none(m2.nano_idx, -1)).astype(np.int64))


def pair_rows(keep, primary_sel, wp_sel, wp_ok, muons):
    """Output rows: row 0 = the primary pair of every kept event, plus one row per WP (in wp_sel order) whose
    leading pair passes wp_ok and differs from every earlier row of that event.

    Returns (row_idx into events, row number within its event, n rows of its event, per-row muon mask).
    """
    keep = np.asarray(ak.to_numpy(ak.fill_none(keep, False)), dtype=bool)
    ev, masks = [np.flatnonzero(keep)], [primary_sel[keep]]
    seen = [_pair_ids(muons, primary_sel)]
    for tag, sel in wp_sel.items():
        a, b = _pair_ids(muons, sel)
        new = keep & (b >= 0) & np.asarray(ak.to_numpy(ak.fill_none(wp_ok[tag], False)), dtype=bool)
        for s1, s2 in seen:
            new &= ~((a == s1) & (b == s2))
        seen.append((a, b))
        ev.append(np.flatnonzero(new))
        masks.append(sel[new])
    ev = np.concatenate(ev)
    order = np.argsort(ev, kind="stable")  # rows of one event adjacent, primary first
    row_idx = ev[order]
    row_num = np.arange(len(row_idx)) - np.searchsorted(row_idx, row_idx, side="left")
    n_rows = np.bincount(row_idx, minlength=len(keep))[row_idx]
    row_sel = ak.to_packed(ak.concatenate(masks, axis=0)[order])
    return row_idx, row_num, n_rows, row_sel


def row_is_pair(muons, row_sel, sel):
    """Per row: the leading pair of sel is the row's pair (the one its mu1/mu2/jet/weight columns use)."""
    r1, r2 = _pair_ids(muons, row_sel)
    w1, w2 = _pair_ids(muons, sel)
    return (w2 >= 0) & (w1 == r1) & (w2 == r2)

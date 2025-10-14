import pickle
import hist
import os
import glob

# Directory with the .pkl files
indir = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_17July//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_17July/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test/2018/"
files = glob.glob(os.path.join(indir, "*.pkl"))

old_prefix = "score_Run2_nanoAODv12_UpdatedQGL_17July" # starts with this
new_axis_name = "score_Run2_nanoAODv12_UpdatedQGL_17July"

for fn in files:
    print(f"Processing: {fn}")
    with open(fn, "rb") as f:
        try:
            h = pickle.load(f)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
    # Check if it's a hist.hist.Hist object and needs renaming
    found = False
    axes = []
    for ax in h.axes:
        if ax.name.startswith(old_prefix):
            axes.append(hist.axis.Variable(ax.edges, name=new_axis_name))
            found = True
        else:
            axes.append(ax)
    if not found:
        print("  No axis renaming needed, skipping.")
        continue
    # Make new hist and copy contents
    new_h = hist.Hist(*axes, storage=h.storage_type())
    new_h.view(flow=True)[...] = h.view(flow=True)
    # Save with a suffix or in a new directory
    outfile_dir = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_17July//stage2_histograms/score_Run2_nanoAODv12_UpdatedQGL_17July/2018_h-peak_vbf_2018_UpdatedQGL_17July_Test_RenameScore/2018/"
    os.makedirs(outfile_dir, exist_ok=True)
    out_fn = os.path.join(outfile_dir, os.path.basename(fn))
    print(f"  Renamed axis and saving to: {out_fn}")
    with open(out_fn, "wb") as fout:
        pickle.dump(new_h, fout)
    print(f"  Saved renamed hist to: {out_fn}")

print("All matching .pkl files processed.")

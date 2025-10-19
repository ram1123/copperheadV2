import ROOT as rt
import yaml
import os

# ROOT Error Handling: Suppress non-critical warnings
rt.gErrorIgnoreLevel = rt.kError  # Only show errors, not warnings

# Enable ROOT's implicit multi-threading for better performance
rt.EnableImplicitMT()

# File paths and plot directory setup
file1_path = "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/*.root"
plots_path = "plots/nanoAODv12/SingleMuon_Run2018C/"
os.makedirs(plots_path, exist_ok=True)  # Ensure the plots directory exists

# Variables to compare
comparison_pairs = [
    ("Muon_pt", "Muon_bsConstrainedPt"),
    ("Muon_ptErr", "Muon_bsConstrainedPtErr")
]

# Define cuts (if any) to be applied during the comparison
cuts = {
    "B": "abs(Muon_eta) < 0.9",  # Barrel region
    "O": "abs(Muon_eta) > 0.9 && abs(Muon_eta) < 1.8",  # Overlap region
    "E": "abs(Muon_eta) > 1.8 && abs(Muon_eta) < 2.4"  # Endcap region
}

# Function to create and customize the histograms
def create_histogram(rdf, variable, hist_name, hist_title, binning, x_title, y_title, line_color):
    var = rdf.Redefine(variable, f"{variable}")
    hist = var.Histo1D((hist_name, hist_title, *binning), variable)
    hist.GetXaxis().SetTitle(x_title)
    hist.GetYaxis().SetTitle(y_title)
    hist.SetLineColor(line_color)
    # return hist.GetPtr()
    return hist

# Load data using RDataFrame
rdf1 = rt.RDataFrame("Events", file1_path)

# Plot properties
binning = (100, 0, 200)
x_title = "Muon pT [GeV]"
y_title = "Entries"

# Loop over cuts (you can add a case for "No Cut" to compare without selection)
for cut_name, cut_condition in cuts.items():
    rdf_cut = rdf1.Filter(cut_condition, cut_name)  # Apply the cut to the RDataFrame
    print(f"Applied cut: {cut_name} with condition: {cut_condition}")

    canvas = rt.TCanvas(f"canvas_{cut_name}", f"{cut_name} Comparison", 800, 600)
    rt.gStyle.SetOptStat(0)  # Hide stats

    legend = rt.TLegend(0.7, 0.7, 0.95, 0.9)
    first_hist = True  # Flag to check if it's the first histogram to be drawn

    # Loop through the variable pairs and create histograms for each
    for var1, var2 in comparison_pairs:
        # Plot first variable
        hist_name = f"h_{var1}_{cut_name}"
        hist_title = f"{var1} for {cut_name}"
        line_color = rt.kRed  # Color for the first variable
        h_var1 = create_histogram(rdf_cut, var1, hist_name, hist_title, binning, x_title, y_title, line_color)

        # Plot second variable
        hist_name = f"h_{var2}_{cut_name}"
        hist_title = f"{var2} for {cut_name}"
        line_color = rt.kBlue  # Color for the second variable
        h_var2 = create_histogram(rdf_cut, var2, hist_name, hist_title, binning, x_title, y_title, line_color)

        # Draw histograms
        if first_hist:
            h_var1.Draw("HIST")
            h_var2.Draw("HIST SAME")
            first_hist = False
        else:
            h_var1.Draw("HIST SAME")
            h_var2.Draw("HIST SAME")

        # Add the legends
        legend.AddEntry(h_var1, f"{var1} {cut_name}", "l")
        legend.AddEntry(h_var2, f"{var2} {cut_name}", "l")

    # Add the legend and save the plot
    legend.Draw()
    canvas.SaveAs(f"{plots_path}/{cut_name}_comparison.pdf")
    print(f"Plot saved for {cut_name}: {plots_path}/{cut_name}_comparison.pdf")

# Clean up
del rdf1, canvas

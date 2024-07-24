import ROOT
import time
 
# Create model for physics sample
# -------------------------------------------------------------
start = time.time()
# Create observables
mass = ROOT.RooRealVar("x", "x", -8, 8)
 
# Construct signal pdf
mean = ROOT.RooRealVar("mean", "mean", 0, -8, 8)
sigma = ROOT.RooRealVar("sigma", "sigma", 0.3, 0.1, 10)
gx = ROOT.RooGaussian("gx", "gx", mass, mean, sigma)
 
# Construct background pdf
a0 = ROOT.RooRealVar("a0", "a0", -0.1, -1, 1)
a1 = ROOT.RooRealVar("a1", "a1", 0.004, -1, 1)
px = ROOT.RooChebychev("px", "px", mass, [a0, a1])
 
# Construct composite pdf
f = ROOT.RooRealVar("f", "f", 0.2, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [gx, px], [f])
 
# Create model for control sample
# --------------------------------------------------------------
 
# Construct signal pdf.
# NOTE that sigma is shared with the signal sample model
mean_ctl = ROOT.RooRealVar("mean_ctl", "mean_ctl", -3, -8, 8)
gx_ctl = ROOT.RooGaussian("gx_ctl", "gx_ctl", mass, mean_ctl, sigma)
 
# Construct the background pdf
a0_ctl = ROOT.RooRealVar("a0_ctl", "a0_ctl", -0.1, -1, 1)
a1_ctl = ROOT.RooRealVar("a1_ctl", "a1_ctl", 0.5, -0.1, 1)
px_ctl = ROOT.RooChebychev("px_ctl", "px_ctl", mass, [a0_ctl, a1_ctl])
 
# Construct the composite model
f_ctl = ROOT.RooRealVar("f_ctl", "f_ctl", 0.5, 0.0, 1.0)
model_ctl = ROOT.RooAddPdf("model_ctl", "model_ctl", [gx_ctl, px_ctl], [f_ctl])
 
# Generate events for both samples
# ---------------------------------------------------------------
 
# Generate 1000 events in x and y from model
data = model.generate({mass}, 1000)
data_ctl = model_ctl.generate({mass}, 2000)
 
# Create index category and join samples
# ---------------------------------------------------------------------------
 
# Define category to distinguish physics and control samples events
sample = ROOT.RooCategory("sample", "sample")
sample.defineType("physics")
sample.defineType("control")
 
# Construct combined dataset in (x,sample)
combData = ROOT.RooDataSet(
    "combData",
    "combined data",
    {mass},
    Index=sample,
    Import={
        "physics": data, 
        "control": data_ctl
    },
)
 
# Construct a simultaneous pdf in (x, sample)
# -----------------------------------------------------------------------------------
 
# Construct a simultaneous pdf using category sample as index: associate model
# with the physics state and model_ctl with the control state
simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", 
                              {
                                  "physics": model, 
                                   "control": model_ctl
                              }, sample
                             )
 
# Perform a simultaneous fit
# ---------------------------------------------------
 
# Perform simultaneous fit of model to data and model_ctl to data_ctl
fitResult = simPdf.fitTo(combData, PrintLevel=-1, Save=True)
fitResult.Print()

end = time.time()
print(f"runtime: {end-start} seconds")
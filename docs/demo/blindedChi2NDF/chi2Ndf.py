import ROOT
from ROOT import RooFit
"""
This python script is to step-by-step verify an implementation
to obtain chi2ndf from a H-sidebands fit, where you have to be blinded
over the H-peak region
"""

def chi2_ndf_manual(pdf, data_hist, x, regions, nfloat):
    chi2 = 0.0
    nbin = 0
    data_hist = data_hist.createHistogram("data_hist", x)
    for i in range(1, data_hist.GetNbinsX()+1):
        xc = data_hist.GetXaxis().GetBinCenter(i)

        # check if bin is in any allowed region
        if not any(lo <= xc <= hi for (lo, hi) in regions):
            continue

        obs = data_hist.GetBinContent(i)
        err = data_hist.GetBinError(i)
        if err <= 0:
            print(f"NOTE: bin {i} has error zero. Skipping!")
            continue

        x.setVal(xc)
        exp = pdf.getVal(ROOT.RooArgSet(x)) * data_hist.Integral() * data_hist.GetBinWidth(i)

        chi2 += (obs - exp)**2 / err**2
        nbin += 1

    # print(f"chi2_ndf_manual data_hist.GetNbinsX(): {data_hist.GetNbinsX()}")
    # print(f"chi2_ndf_manual nbin: {nbin}")
    ndf = nbin - nfloat
    return chi2, ndf

# ------------------------------
# Step1: Test different chi2/ndf methods
# over simulated gaussian distribution
# ------------------------------

# Observable
xmin = -3
xmax = 3
x = ROOT.RooRealVar("x", "x", xmin, xmax)

# Define binning explicitly (important!)
nbins=100
x.setBins(nbins)

# Gaussian PDF
mean  = ROOT.RooRealVar("mean",  "mean",  0, -5, 5)
sigma = ROOT.RooRealVar("sigma", "sigma", 1,  0.1, 5)
pdf   = ROOT.RooGaussian("pdf", "pdf", x, mean, sigma)
nFit_params = 2
# Generate unbinned data
data = pdf.generate(ROOT.RooArgSet(x), 10_000_000)

# Convert to binned dataset
datahist = ROOT.RooDataHist(
    "datahist",
    "datahist",
    ROOT.RooArgSet(x),
    data
)

# Fit to binned data
# pdf.fitTo(datahist, ROOT.RooFit.PrintLevel(-1))
fitresult = pdf.chi2FitTo(datahist, ROOT.RooFit.PrintLevel(-1), Save=True)
# fitresult = pdf.fitTo(datahist, ROOT.RooFit.PrintLevel(-1), Save=True)

fitresult.Print("v")

# ------------------------------
# method 1
# extract chi2/ndf from the frame NOTE: "the chi2 is calculated between a plot of the original distribution and the data. It therefore has more rounding errors " from https://root.cern.ch/doc/master/classRooPlot.html#ae2bbe62ce38dbf09bc894acbb117e68d
# ------------------------------
frame = x.frame()
datahist.plotOn(frame, Name="datahist")
pdf.plotOn(frame, Name="pdf")
chi2ndf = frame.chiSquare("pdf", "datahist", nFit_params) # chiSquare demo source: https://root.cern/doc/v638/rf109__chi2residpull_8py.html
print(f"step1 method 1 chi2ndf = {chi2ndf}")
# step1 method 1 chi2ndf = 1.2253893422664321

# ------------------------------
# method 2
# extract chi2NDF by directly extracting chi2 and calculating it
# ------------------------------
chi2_obj = pdf.createChi2( # createChi2 demo: https://root.cern/doc/v638/rf602__chi2fit_8py.html
    datahist,
)
chi2 = chi2_obj.getVal()
NDF = nbins - nFit_params
chi2ndf = chi2/NDF
print(f"step1 method 2 chi2 = {chi2}")
print(f"step1 method 2 NDF = {NDF}")
print(f"step1 method 2 chi2ndf = {chi2ndf}")
# print output:
# step1 method 2 chi2 = 116.396219174843
# step1 method 2 NDF = 98
# step1 method 2 chi2ndf = 1.1877165221922754


# ------------------------------
# method 3
# extract chi2NDF by manually caluculating chi2 bin by bin
# comparing histogram and fit value
# ------------------------------
regions = [(xmin,xmax)]
chi2, NDF = chi2_ndf_manual(pdf, datahist, x, regions, nFit_params)
print(f"step1 method 3 chi2 = {chi2}")
print(f"step1 method 3 NDF = {NDF}")
print(f"step1 method 3 chi2ndf = {chi2/NDF}")
# print output:
# step1 method 3 chi2 = 116.52038967509172
# step1 method 3 NDF = 98
# step1 method 3 chi2ndf = 1.1889835681131808

"""
# NOTE: the difference chi2ndf values between method1 and method2 is about 3%, 
the difference chi2ndf values between method2 and method3 is about 0.1%. 
Conclusion: When calculating over one continuous region, all methods work fine
"""


# ------------------------------
# Step2: Test different chi2/ndf methods over
# smoothely decaying distribution defined by
# sumExponential function
# ------------------------------
# define new x
x_name = "mh_ggh"
x = ROOT.RooRealVar(x_name, x_name, 120, 110, 150)
nbins = 800
x.setBins(nbins)
x.setRange("full", 110, 150 )

# define new pdf
name = f"RooSumTwoExpPdf_a1_coeff"
a1_coeff = ROOT.RooRealVar(name,name, -1.4756e-01,-2.0,1)
name = f"RooSumTwoExpPdf_a2_coeff"
a2_coeff = ROOT.RooRealVar(name,name, -3.4552e-02,-2.0,1)
name = f"RooSumTwoExpPdf_f_coeff"
f_coeff = ROOT.RooRealVar(name,name,  2.4864e-01,0.0,1.0)
pdf = ROOT.RooSumTwoExpPdf("RooSumTwoExpPdf", "RooSumTwoExpPdf", x, a1_coeff, a2_coeff, f_coeff) 
nFit_params = 3

# Generate unbinned data
data = pdf.generate(ROOT.RooArgSet(x), 10_000_000)
# Convert to binned dataset
datahist = ROOT.RooDataHist(
    "datahist",
    "datahist",
    ROOT.RooArgSet(x),
    data
)

# Divide binned dataset over hiHalf and loHalf
x.setRange("hiHalf", 125, 150 )
x.setRange("loHalf", 110, 125 )
data_hiHalf = data.reduce(RooFit.CutRange("hiHalf"))
datahist_hiHalf = ROOT.RooDataHist(
    "datahist_hiHalf",
    "datahist_hiHalf",
    ROOT.RooArgSet(x),
    data_hiHalf
)
data_loHalf = data.reduce(RooFit.CutRange("loHalf"))
datahist_loHalf = ROOT.RooDataHist(
    "datahist_loHalf",
    "datahist_loHalf",
    ROOT.RooArgSet(x),
    data_loHalf
)

# Verify the result
# ------------------------------
# print(f"data_hiHalf.numEntries(): {data_hiHalf.numEntries()}")
# print(f"Original dataset entries: {datahist.sumEntries()}")
# print(f"datahist_hiHalf entries: {datahist_hiHalf.sumEntries()}")
# print(f"datahist_loHalf entries: {datahist_loHalf.sumEntries()}")
# ------------------------------


fitresult = pdf.chi2FitTo(datahist, ROOT.RooFit.PrintLevel(-1), Save=True)
fitresult.Print("v")

# ------------------------------
# method 2
# extract chi2NDF by directly extracting chi2 and calculating it
# ------------------------------
chi2_obj = pdf.createChi2( # createChi2 demo: https://root.cern/doc/v638/rf602__chi2fit_8py.html
    datahist,
)
chi2 = chi2_obj.getVal()
NDF = nbins - nFit_params
chi2ndf = chi2/NDF
print(f"step2 method 2 chi2 = {chi2}")
print(f"step2 method 2 NDF = {NDF}")
print(f"step2 method 2 chi2ndf = {chi2ndf}")
# step2 method 2 chi2 = 749.5105063256887
# step2 method 2 NDF = 797
# step2 method 2 chi2ndf = 0.9404146879870623

# ------------------------------
# method 3
# extract chi2 individually from loHalf and hiHalf, add them up
# and divide them by ndf
# ------------------------------
chi2_hiHalf_obj = pdf.createChi2( # createChi2 demo: https://root.cern/doc/v638/rf602__chi2fit_8py.html
    datahist_hiHalf,
    RooFit.CutRange("hiHalf"),
)
chi2_loHalf_obj = pdf.createChi2( # createChi2 demo: https://root.cern/doc/v638/rf602__chi2fit_8py.html
    datahist_loHalf,
    RooFit.CutRange("loHalf"),
)
chi2 = chi2_hiHalf_obj.getVal() + chi2_loHalf_obj.getVal()
NDF = nbins - nFit_params
chi2ndf = chi2/NDF
print(f"step2 method 3 chi2_hiHalf_obj.getVal() = {chi2_hiHalf_obj.getVal()}")
print(f"step2 method 3 chi2_loHalf_obj.getVal() = {chi2_loHalf_obj.getVal()}")
print(f"step2 method 3 chi2 = {chi2}")
print(f"step2 method 3 NDF = {NDF}")
print(f"step2 method 3 chi2ndf = {chi2ndf}")
# print output:
# step2 method 3 chi2_hiHalf_obj.getVal() = 6280176.725606403
# step2 method 3 chi2_loHalf_obj.getVal() = 3721066.8016281947
# step2 method 3 chi2 = 10001243.527234599
# step2 method 3 NDF = 797
# step2 method 3 chi2ndf = 12548.611702929233


# visualizing method 3 to help understanding the crazy values
# ------------------------------
# frame = x.frame()
# datahist_hiHalf.plotOn(frame, Name="datahist_hiHalf", MarkerColor=ROOT.kGreen)
# datahist_loHalf.plotOn(frame, Name="datahist_loHalf", MarkerColor=ROOT.kBlue)
# c = ROOT.TCanvas("rf101_basics", "rf101_basics", 800, 400)
# frame.Draw()
# c.SaveAs("step2_method3Visualization.png")
# ------------------------------


regions = [(110,125), (125,150)]
chi2, NDF = chi2_ndf_manual(pdf, datahist, x, regions, nFit_params)
print(f"step2 method 4 chi2 = {chi2}")
print(f"step2 method 4 NDF = {NDF}")
print(f"step2 method 4 chi2ndf = {chi2/NDF}")
# print output:
# step2 method 4 chi2 = 750.1401664386382
# step2 method 4 ndf = 797
# step2 method 4 chi2ndf = 0.941204725769935

"""
# NOTE: we verify that chi2 obtained from chi2_ndf matches with method2. However, method3 doesn't work when it comes to combining two regions. This is an implementation issue, but I couldn't find a workaround.
thus we don't
"""

# ------------------------------
# Step3: We use method 4 to obtain the chi2/ndf from sideband region only,
# which is what is needed for blinded analysis. We expect the chi2/ndf to be
# similar to chi2/ndf over full signal region
# ------------------------------
# set sideband range
x.setRange("hiSB", 135, 150 )
x.setRange("loSB", 110, 115 )
fit_range = "loSB,hiSB"


# to ensure no bias from the previous full region fit, re-initialize pdf
name = f"RooSumTwoExpPdf_a1_coeff"
a1_coeff = ROOT.RooRealVar(name,name, -1.4756e-01,-2.0,1)
name = f"RooSumTwoExpPdf_a2_coeff"
a2_coeff = ROOT.RooRealVar(name,name, -3.4552e-02,-2.0,1)
name = f"RooSumTwoExpPdf_f_coeff"
f_coeff = ROOT.RooRealVar(name,name,  2.4864e-01,0.0,1.0)
pdf = ROOT.RooSumTwoExpPdf("RooSumTwoExpPdf", "RooSumTwoExpPdf", x, a1_coeff, a2_coeff, f_coeff) 
# fit
fitresult = pdf.chi2FitTo(datahist, RooFit.Range(fit_range), ROOT.RooFit.PrintLevel(-1), Save=True)
fitresult.Print("v")

regions = [(110,115), (135,150)]
chi2, NDF = chi2_ndf_manual(pdf, datahist, x, regions, nFit_params)
print(f"step3 method 4 chi2 = {chi2}")
print(f"step3 method 4 NDF = {NDF}")
print(f"step3 method 4 chi2ndf = {chi2/NDF}")
# step3 method 4 chi2 = 336.65461935533114
# step3 method 4 ndf = 397
# step3 method 4 chi2ndf = 0.84799652230562
"""
# NOTE: we observer that chi2 obtained from chi2_ndf over the blinded region
(sideband region)(chi2ndf=0.848) and is comparable to to that over the full
signal region (chi2ndf=0.941) (about 10% difference).
Conclusion: chi2_ndf_manual() can be used to calculate the chi2ndf for fit 
functions over sideband regions.
"""



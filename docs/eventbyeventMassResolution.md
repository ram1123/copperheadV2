# Basic Information
<!--
Z mass: 91.1880 GeV from PDG -->
<!-- Natural Z width 2.4955 GeV from PDG -->

- Z boson mass: 91.1880 GeV (from PDG)
- Natural Z width: 2.4955 GeV (from PDG)
- H boson mass: 125.200 GeV (from PDG)
- H boson width: 3.7 MeV (from PDG)


# Fitting of Z-peak Mass Resolution

## Things to note

1. DCB convoluted with BW function is not same as BW convoluted with DCB function.
2. Fit Z-peak with BW convoluted with DCB function.
3. Fit H-peak with DCB funtion

## RooFit Information

- If using GPU, then setup environment using:

   ```bash
   source /cvmfs/sft.cern.ch/lcg/views/LCG_106b_cuda/x86_64-el8-gcc11-opt/setup.sh
   ```

- To get the chi2/NDF use this method:

   ```python
   new_nfree_params = fit_result.floatParsFinal().getSize()
   chi2_ndf = frame.chiSquare("model_bsOn", "hist_bsOn", new_nfree_params)
   ```

- To get the fit result use this method:



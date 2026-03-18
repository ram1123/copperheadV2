---
title: Event-by-Event Mass Resolution Calibration
---

# Event-by-Event Mass Resolution Calibration

This document describes the **event-by-event (EBE) dimuon mass resolution calibration** framework used in the analysis.
The purpose of this calibration is to ensure that the **predicted per-event dimuon mass resolution** correctly represents the **observed mass resolution** in data, as measured from the **Z -> μμ** resonance.

The calibration is derived in the **Z control region (ZCR)** and then applied consistently to the higgs signal region and higgs sideband regions.


# Physics Inputs To the Calibration

The following physics constants are used for initializing the functions modeling:

- **Z boson mass**: 91.1880 GeV (PDG)
- **Natural Z width**: 2.4955 GeV (PDG)
- **H boson mass**: 125.200 GeV (PDG)
- **H boson width**: 3.7 MeV (PDG)



# What Is the Event-by-Event Mass Resolution?

For each event, the dimuon mass resolution is **predicted** using the transverse momentum uncertainties of the two muons:

$$
\sigma_{m_{\mu\mu}}^{\text{pred}} =
\frac{m_{\mu\mu}}{2}
\sqrt{
\left(\frac{\Delta p_T(\mu_1)}{p_T(\mu_1)}\right)^2
+
\left(\frac{\Delta p_T(\mu_2)}{p_T(\mu_2)}\right)^2
}
$$

where:

- $ m_{\mu\mu} $ is the reconstructed dimuon mass,
- $ p_T(\mu_i) $ is the transverse momentum of muon $ i $,
- $ \Delta p_T(\mu_i) $ is the estimated uncertainty on the muon $ p_T $.

This quantity is computed **event-by-event** and is referred to as the **predicted (Non-calibrated) mass resolution**.

---

# Why Is Calibration Needed?

Although the predicted resolution captures the main detector effects, it is not perfectly modeled in data due to:

- Imperfect modeling of muon momentum uncertainties
- Detector non-idealities
- Residual mismodeling in simulation and reconstruction

As a result, the predicted resolution must be **scaled** to match the **observed resolution** extracted from the Z mass peak.


$$
\sigma_{m_{\mu\mu}}^{\text{calibrated}} =
\text{CalibrationFactor} \times
\sigma_{m_{\mu\mu}}^{\text{pred}}
$$

The calibration factor is derived in bins of muon kinematics (both in $ p_T $ and $ \eta $) to account for variations in detector performance. In this way, the calibrated resolution better reflects the true mass resolution observed in data.

**The calibration factor is computed as the ratio between the resolution extracted from the Z peak fit and the median of the predicted resolution in the same category.**

$$
\text{CalibrationFactor} =
\frac{\sigma_{\text{fit}}}{\text{median}(\sigma_{m_{\mu\mu}}^{\text{pred}})}
$$

# Calibration Strategy

The calibration proceeds as follows:

1. Select **Z -> μμ events** in the Z control region (ZCR)
2. Split events into **analysis categories**, based on transverse momentum and pseudorapidity of muons
3. For each category:
   - Fit the Z mass peak
   - Extract the **effective mass resolution** from the fit
4. Compute the **median predicted resolution** for events in the same category
5. Define the calibration factor as:
   $$
   \text{CalibrationFactor} =
   \frac{\sigma_{\text{fit}}}{\text{median}(\sigma_{m_{\mu\mu}}^{\text{pred}})}
   $$
6. Store calibration factors in a **JSON correction file**
7. Apply the calibration during Stage-1 processing
8. Validate the procedure using **closure tests**


## Script and Location

The calibration is implemented in the script:

```text
src/lib/ebeMassResCalibration/getCalibrationFactor.py
```

This script:
- Reads Stage-1 compacted parquet files
- Performs Z-peak fits
- Computes calibration factors
- Produces diagnostic plots and summary CSV/JSON outputs


## How to Run

Before running the calibration, **verify and update the input parquet paths** inside:

```text
src/lib/ebeMassResCalibration/getCalibrationFactor.py
```

These paths must point to the correct **Stage-1 compacted outputs**.


## Standard Calibration Run

```bash
bash stage1_loop_Improved.sh \
  -v <NanoAODv> \
  -y <year> \
  -k
```

Option breakdown:
- `-v <NanoAODv>`: Specify the NanoAOD version (e.g., 12)
- `-y <year>`: Specify the data-taking year (e.g., 2018)
- `-k`: Enable dask gateway for distributed processing

This command:
- Runs Stage-1 in calibration mode
- Produces a JSON file with EBE mass resolution calibration factors

***Example of command variations:***

1. Calibrate data:

    ```bash
    python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv $NanoAODv --years $year --extraString V1 --ifbinned --steps all
    ```

2. Calibrate MC:

    ```bash
    python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv $NanoAODv --years $year --extraString V1 --ifbinned --steps all --isMC
    ```

3. Calibrate a specific category (e.g., category 30-45_OB):

    For some category (say 30-45 OB), if the fit it not good, then we should just re-run the fit for that category only, after tuning the fit options. For that add the `--fixCat` option:


    ```bash
    python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv $NanoAODv --years $year --extraString V1 --ifbinned --steps step1  --fixCat ${category}
    ```

4. Closure test on data:

    ```bash
    python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv $NanoAODv --years $year --extraString V1 --ifbinned --closure_test
    ```

5. Closure test on MC:

    ```bash
    python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv $NanoAODv --years $year --extraString V1 --ifbinned --closure_test --isMC
    ```

6. Closure test for a specific category (e.g., category 30-45_OB):

    ```bash
    python src/lib/ebeMassResCalibration/getCalibrationFactor.py --nanoAODv $NanoAODv --years $year --extraString V1 --ifbinned --closure_test  --fixCat ${category} --no-dask-client
    ```

## Using the Calibration Output

After the calibration completes:

1. Copy the generated JSON file to:
   ```text
   data/res_calib/
   ```

2. Update the correction list in:
   ```text
   configs/parameters/correction_filelist.yaml
   ```

3. Re-run **Stage-1 processing** with the calibrated resolution enabled.

> **Important:**
Ensure that the **BSC / EBE resolution calibration option is switched ON** during Stage-1.


## Closure Test

### Purpose

The closure test verifies that:
- The calibrated predicted resolution
- Correctly reproduces the fitted Z mass width
- Across the full resolution spectrum

To avoid bias from the nominal analysis categories, the closure test uses an **independent binning in predicted resolution**.


### Closure Bin Definition

```python
CLOSURE_BINS = [
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
    (1.3, 1.4),
    (1.4, 1.5),
    (1.5, 1.7),
    (1.7, 2.0),
    (2.0, 2.5),
    (2.5, 3.5),
]
```

Each bin corresponds to a range in **predicted dimuon mass resolution (GeV)**.


### Running the Closure Test

```bash
python src/lib/ebeMassResCalibration/getCalibrationFactor.py \
  --nanoAODv 12 \
  --years "2018" \
  --ifbinned \
  --closure_test
```

This will:
- Apply the calibration
- Refit the Z peak in each resolution bin
- Compare fitted vs predicted resolutions
- Produce validation plots and CSV summaries


## RooFit Technical Notes

### Optional GPU Support

If running RooFit with GPU support:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106b_cuda/x86_64-el8-gcc11-opt/setup.sh
```


### Computing χ² / NDF

To compute χ² per degree of freedom consistently:

```python
n_free_params = fit_result.floatParsFinal().getSize()
chi2_ndf = frame.chiSquare("model_bsOn", "hist_bsOn", n_free_params)
```

This method ensures:
- Correct handling of floating parameters
- Comparable fit-quality metrics across categories


## Modeling Guidelines and Caveats

1. **Convolution order matters**
   - `DCB ⊗ BW ≠ BW ⊗ DCB`
   - Always use **BW ⊗ DCB** for Z-peak fits

2. **Z-peak modeling**
   - Breit–Wigner ⊗ Double Crystal Ball
   - Natural width must be included

3. **Higgs-peak modeling**
   - Pure Double Crystal Ball
   - Higgs natural width is negligible compared to detector resolution


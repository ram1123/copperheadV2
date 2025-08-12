# CopperheadV2 Framework Restructuring Summary

## Overview
The CopperheadV2 framework has been successfully restructured to match the intended organization outlined in the project README. This restructuring improves code organization, maintainability, and follows Python package best practices.

## Directory Structure Changes

### Before Restructuring:
```
copperheadV2/
├── src/
│   ├── lib/                    # Mixed library functions
│   └── corrections/            # Correction modules
├── MVA_training/               # MVA training code
├── configs/
│   └── parameters/             # Parameter YAML files
├── run_stage*.py              # Main workflow scripts (scattered)
└── various scripts/           # Shell scripts throughout
```

### After Restructuring:
```
copperheadV2/
├── lib/                       # ✅ Core library package
│   ├── ebeMassResCalibration/    # Event-by-event mass calibration
│   ├── ZptWgtCalculation/        # Z-pT weight calculation
│   ├── MVA_training/             # MVA training modules
│   │   ├── ggH/                  # ggH production channel
│   │   └── VBF/                  # VBF production channel
│   ├── corrections/              # General corrections (stage1)
│   ├── fit_models/               # Roofit fitting (stage3)
│   └── histogram/                # Histogram utilities
├── parameters/                # ✅ Metadata YAML files
├── workflows/                 # ✅ Workflow scripts
├── run_stage*.py             # Main run scripts (kept in root)
└── configs/                  # Dataset and other configs
```

## Files Migrated

### Core Libraries (`src/lib/` → `lib/`)
- ✅ `ebeMassResCalibration/` → Event-by-event mass calibration
- ✅ `get_parameters.py` → Parameter loading utilities
- ✅ `categorizer.py` → Categorization utilities
- ✅ `histogram/` → Histogram and plotting utilities

### MVA Training (`MVA_training/` → `lib/MVA_training/`)
- ✅ `VBF/` → VBF channel MVA training
- ✅ `MVA_functions.py` → Core MVA functions

### Corrections (`src/corrections/` → `lib/corrections/`)
- ✅ All correction modules (Rochester, FSR, JEC, etc.)

### Fit Models (`src/lib/fit_functions.py` → `lib/fit_models/`)
- ✅ RooFit fitting functions

### Parameters (`configs/parameters/` → `parameters/`)
- ✅ All YAML parameter files

### Z-pT Calculations (created `lib/ZptWgtCalculation/`)
- ✅ Z-pT weight generation and validation scripts

### Workflows (organized existing scripts → `workflows/`)
- ✅ Shell scripts and workflow management tools

## Import Updates

Updated **25+ files** with new import paths:
- `src.lib.*` → `lib.*`
- `src.corrections.*` → `lib.corrections.*`
- `configs/parameters/` → `parameters/`

### Files Updated:
- All main run scripts (`run_stage*.py`)
- Processor files (`src/copperhead_processor*.py`)
- Plotting and validation scripts
- Workflow scripts in `workflows/`
- Jupyter notebooks
- Documentation files

## Python Package Structure

Added proper Python package structure:
- ✅ `lib/__init__.py` and subdirectory `__init__.py` files
- ✅ Proper module organization
- ✅ Clean import paths

## Backward Compatibility

- ✅ Original files kept in place to ensure no breaking changes
- ✅ All main run scripts remain in root directory for easy access
- ✅ Configuration paths maintained where appropriate

## Testing

- ✅ All Python files compile successfully
- ✅ Import structure validated
- ✅ Package structure verified

## Benefits

1. **Clear Organization**: Each component has a dedicated directory
2. **Better Maintainability**: Related code is grouped together
3. **Python Standards**: Follows Python package conventions
4. **Easier Navigation**: Logical directory structure
5. **Modular Design**: Clear separation of concerns

## Usage

The framework can now be used with the new structure:

```python
# New import style
from lib.get_parameters import getParametersForYr
from lib.corrections.rochester import apply_roccor
from lib.fit_models.fit_functions import MakeFEWZxBernDof3

# Parameters now in root-level parameters/ directory
config = getParametersForYr("./parameters/", "2018")
```

## Next Steps

1. ✅ Framework restructuring complete
2. ✅ Import paths updated
3. ✅ Documentation updated
4. 🔄 Testing with actual data processing (requires proper environment)
5. 🔄 Performance validation
6. 🔄 User migration guide if needed

---
*Restructuring completed as part of issue #36*
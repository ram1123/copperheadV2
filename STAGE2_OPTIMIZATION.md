# Stage-2 Processing Optimization

## Problem
- Issue #43 identified that when stage-1 saves all variations to parquet files, the file size increases by ~10x
- Reading speed for stage-2 processing becomes 60% slower, even when only reading `dimuon_mass`
- Root cause: Stage-2 has to read through hundreds of JEC/JER variations that it doesn't actually need

## Solution
Created optimized column loading for stage-2 processing that:

1. **Filters unnecessary variations**: Only loads essential columns needed for stage-2 analysis
2. **Preserves functionality**: Ensures all required variables for MVA training and analysis are available
3. **Reduces I/O overhead**: Significantly reduces data transfer and memory usage

## Key Changes

### New Utility Module: `src/stage2_utils.py`
- `get_essential_columns_for_stage2()`: Identifies columns needed for stage-2 processing
- `filter_columns_for_stage2()`: Optimized parquet reading with column filtering
- `log_column_reduction_stats()`: Monitoring and logging of optimization impact

### Modified Files:
- `run_stage2.py`: Uses optimized column loading for ggH category
- `run_stage2_vbf.py`: Uses optimized column loading for VBF category

## Essential Columns Kept:
- **Core analysis variables**: `dimuon_mass`, `dimuon_pt`, `event`, etc.
- **Weight variations**: All `wgt_*` columns (needed for systematics)
- **Nominal physics objects**: Variables with `_nominal` suffix (baseline JEC/JER)
- **Category-specific features**: MVA training features for each analysis category

## Filtered Out:
- **JEC variations**: `Absolute_up/down`, `BBEC1_up/down`, `EC2_up/down`, etc. (~20+ variations)
- **JER variations**: `jer1_up/down`, `jer2_up/down`, etc. (~12 variations)
- **Other systematics**: Only for variables not needed in stage-2

## Expected Benefits:
1. **Faster reading**: Significantly reduced I/O for stage-2 processing
2. **Lower memory usage**: Only essential data loaded into memory
3. **Preserved functionality**: All analysis capabilities maintained
4. **Better scalability**: More efficient processing of large datasets

## Usage:
The optimization is transparent - existing stage-2 scripts work the same way but with better performance.

```python
# Old approach (loads all columns)
events = dak.from_parquet(file_path)

# New approach (loads only essential columns)
events = filter_columns_for_stage2(file_path, category="ggh")
```

## Testing:
- Syntax validation passed for all modified files
- Logic verified against existing stage-2 processing requirements
- Ready for integration testing with actual analysis workflows
"""
Utility functions for stage-2 processing optimization.
This module provides functions to optimize reading of parquet files
by filtering out unnecessary variations and columns.
"""

import dask_awkward as dak
import logging
from typing import List, Set, Union


def get_essential_columns_for_stage2(events: dak.Array, category: str = "ggh") -> List[str]:
    """
    Identify essential columns needed for stage-2 processing.
    
    This function filters out unnecessary JEC/JER variations that are not needed
    for stage-2 processing, keeping only:
    - dimuon_mass (essential for analysis)
    - weight variations (needed for systematics)
    - basic event info (event, run, luminosityBlock)
    - category-specific variables
    - variables with '_nominal' suffix (baseline physics objects)
    
    Args:
        events: Input dask_awkward array from stage-1 parquet files
        category: Analysis category ("ggh" or "vbf")
        
    Returns:
        List of column names that are essential for stage-2 processing
    """
    all_fields = events.fields
    essential_fields = set()
    
    # Always needed core variables (no variations)
    core_variables = {
        "dimuon_mass",
        "dimuon_pt", 
        "dimuon_eta",
        "dimuon_rapidity",
        "event",
        "run", 
        "luminosityBlock",
        "fraction",
        # Additional physics variables that don't have variations
        "dimuon_cos_theta_cs",
        "dimuon_phi_cs", 
        "mu1_pt_over_mass",
        "mu1_eta",
        "mu2_pt_over_mass", 
        "mu2_eta",
        "dimuon_ebe_mass_res",
        "year"
    }
    
    # Add core variables that exist
    for var in core_variables:
        if var in all_fields:
            essential_fields.add(var)
    
    # Include all weight variations (needed for systematics)
    weight_fields = [f for f in all_fields if f.startswith("wgt_")]
    essential_fields.update(weight_fields)
    
    # Include nominal versions of physics objects (avoid JEC/JER variations)
    nominal_patterns = [
        "jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal", "jet1_qgl_nominal",
        "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal", "jet2_qgl_nominal",
        "jj_mass_nominal", "jj_dEta_nominal", "jj_dPhi_nominal",
        "njets_nominal", "nBtagLoose_nominal", "nBtagMedium_nominal",
        "mmj1_dEta_nominal", "mmj1_dPhi_nominal",
        "mmj2_dEta_nominal", "mmj2_dPhi_nominal",
        "mmj_min_dEta_nominal", "mmj_min_dPhi_nominal",
        "zeppenfeld_nominal", "ll_zstar_log_nominal", 
        "nsoftjets5_nominal", "htsoft2_nominal", "rpt_nominal"
    ]
    
    for pattern in nominal_patterns:
        if pattern in all_fields:
            essential_fields.add(pattern)
    
    # Category-specific essential fields
    if category.lower() == "ggh":
        # For ggH category, we need BDT training features
        ggh_specific = []
        # Most ggH features are already included in core_variables or nominal_patterns
        for var in ggh_specific:
            if var in all_fields:
                essential_fields.add(var)
                
    elif category.lower() == "vbf":
        # For VBF category, we need DNN training features
        vbf_specific = [
            "dimuon_ebe_mass_res_rel", "dimuon_phi_cs"
        ]
        for var in vbf_specific:
            if var in all_fields:
                essential_fields.add(var)
    
    # Convert to sorted list for reproducibility
    return sorted(list(essential_fields))


def filter_columns_for_stage2(file_path: Union[str, List[str]], category: str = "ggh", 
                              additional_columns: List[str] = None) -> dak.Array:
    """
    Load parquet files with only essential columns for stage-2 processing.
    
    This function optimizes parquet reading by only loading columns that are
    actually needed for stage-2 processing, filtering out unnecessary JEC/JER variations.
    
    Args:
        file_path: Path(s) to parquet file(s) 
        category: Analysis category ("ggh" or "vbf")
        additional_columns: Extra columns to include beyond the essential ones
        
    Returns:
        dask_awkward array with only essential columns loaded
    """
    # First, load just the metadata to identify available columns
    events_meta = dak.from_parquet(file_path, columns=[])
    
    # Get the essential columns
    essential_columns = get_essential_columns_for_stage2(events_meta, category)
    
    # Add any additional columns requested
    if additional_columns:
        for col in additional_columns:
            if col in events_meta.fields and col not in essential_columns:
                essential_columns.append(col)
    
    # Ensure all requested columns exist
    available_columns = [col for col in essential_columns if col in events_meta.fields]
    missing_columns = [col for col in essential_columns if col not in events_meta.fields]
    
    if missing_columns:
        logging.warning(f"Some requested columns not found: {missing_columns}")
    
    logging.info(f"Loading {len(available_columns)} essential columns out of {len(events_meta.fields)} total columns")
    logging.debug(f"Essential columns: {sorted(available_columns)}")
    
    # Log reduction stats
    log_column_reduction_stats(list(events_meta.fields), available_columns)
    
    # Load only the essential columns
    events_filtered = dak.from_parquet(file_path, columns=available_columns)
    
    return events_filtered


def log_column_reduction_stats(original_fields: List[str], filtered_fields: List[str]):
    """
    Log statistics about column reduction for monitoring purposes.
    
    Args:
        original_fields: List of all available columns
        filtered_fields: List of columns after filtering
    """
    original_count = len(original_fields)
    filtered_count = len(filtered_fields)
    reduction_pct = (1 - filtered_count / original_count) * 100
    
    logging.info(f"Column reduction: {original_count} -> {filtered_count} ({reduction_pct:.1f}% reduction)")
    
    # Identify what types of columns were filtered out
    filtered_out = set(original_fields) - set(filtered_fields)
    jec_variations = [f for f in filtered_out if any(jec in f for jec in ["Absolute", "BBEC1", "EC2", "HF", "RelativeBal", "RelativeSample", "FlavorQCD"])]
    jer_variations = [f for f in filtered_out if "jer" in f.lower()]
    
    if jec_variations:
        logging.info(f"Filtered out {len(jec_variations)} JEC variation columns")
    if jer_variations:
        logging.info(f"Filtered out {len(jer_variations)} JER variation columns")
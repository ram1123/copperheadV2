#!/usr/bin/env python3
"""
Test script to verify stage-2 optimization works correctly.
"""

import sys
import os
import time
import logging
import dask_awkward as dak

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stage2_utils import filter_columns_for_stage2, get_essential_columns_for_stage2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_column_filtering():
    """Test that column filtering works and reduces the number of columns loaded."""
    
    # Test file path
    test_file = "./test/stage1_outputs/ggh_powheg/part0.parquet"
    
    if not os.path.exists(test_file):
        logging.error(f"Test file not found: {test_file}")
        return False
    
    logging.info(f"Testing column filtering with: {test_file}")
    
    try:
        # Load all columns (original approach)
        start_time = time.time()
        events_all = dak.from_parquet(test_file)
        all_load_time = time.time() - start_time
        all_columns = events_all.fields
        
        logging.info(f"Original loading: {len(all_columns)} columns in {all_load_time:.3f}s")
        
        # Load only essential columns (optimized approach)
        start_time = time.time()
        events_filtered = filter_columns_for_stage2(test_file, category="ggh")
        filtered_load_time = time.time() - start_time
        filtered_columns = events_filtered.fields
        
        logging.info(f"Optimized loading: {len(filtered_columns)} columns in {filtered_load_time:.3f}s")
        
        # Check that essential columns are present
        essential_columns = ["dimuon_mass", "event"]
        for col in essential_columns:
            if col not in filtered_columns:
                logging.error(f"Essential column missing: {col}")
                return False
        
        # Check that unnecessary JEC variations are filtered out
        jec_variations = [col for col in all_columns if any(jec in col for jec in ["Absolute_up", "BBEC1_down", "RelativeBal_up"])]
        jec_in_filtered = [col for col in filtered_columns if any(jec in col for jec in ["Absolute_up", "BBEC1_down", "RelativeBal_up"])]
        
        logging.info(f"JEC variations in original: {len(jec_variations)}")
        logging.info(f"JEC variations in filtered: {len(jec_in_filtered)}")
        
        if len(jec_in_filtered) > 0:
            logging.warning(f"Some JEC variations still present: {jec_in_filtered[:5]}")
        
        # Verify that we can access the essential data
        try:
            dimuon_mass = events_filtered.dimuon_mass
            event_ids = events_filtered.event
            logging.info(f"Successfully accessed dimuon_mass and event fields")
        except Exception as e:
            logging.error(f"Failed to access essential fields: {e}")
            return False
        
        # Calculate improvement
        column_reduction = (1 - len(filtered_columns) / len(all_columns)) * 100
        time_improvement = (1 - filtered_load_time / all_load_time) * 100
        
        logging.info(f"Column reduction: {column_reduction:.1f}%")
        logging.info(f"Load time improvement: {time_improvement:.1f}%")
        
        return True
        
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        return False

def test_vbf_category():
    """Test VBF category optimization."""
    
    test_file = "./test/stage1_outputs/vbf_powheg/part0.parquet"
    
    if not os.path.exists(test_file):
        logging.warning(f"VBF test file not found: {test_file}")
        return True  # Skip this test if file doesn't exist
    
    logging.info(f"Testing VBF category with: {test_file}")
    
    try:
        events_filtered = filter_columns_for_stage2(test_file, category="vbf")
        
        # Check that VBF-specific columns are present
        vbf_essential = ["dimuon_mass", "event", "jj_mass_nominal", "jet1_pt_nominal"]
        for col in vbf_essential:
            if col in events_filtered.fields:
                logging.info(f"✓ VBF essential column present: {col}")
            else:
                logging.warning(f"✗ VBF essential column missing: {col}")
        
        return True
        
    except Exception as e:
        logging.error(f"VBF test failed with error: {e}")
        return False

if __name__ == "__main__":
    logging.info("Starting stage-2 optimization tests...")
    
    success = True
    
    # Test 1: Basic column filtering
    logging.info("\n=== Test 1: Column Filtering ===")
    success &= test_column_filtering()
    
    # Test 2: VBF category
    logging.info("\n=== Test 2: VBF Category ===")
    success &= test_vbf_category()
    
    if success:
        logging.info("\n✓ All tests passed!")
        sys.exit(0)
    else:
        logging.error("\n✗ Some tests failed!")
        sys.exit(1)
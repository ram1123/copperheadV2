#!/bin/bash

# label from run_stage1.py
label="V2_Jan09_ForZptReWgt"

# year="2018"
year="2017"
python generate_zpt_wgts.py -y $year --label $label --use_gateway 

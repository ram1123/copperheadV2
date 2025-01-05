#!/bin/bash
category="vbf"
label="Dec30_vbfCat_HingeLoss"

python dnn_preprocessor.py --label $label  -cat $category
python dnn_train.py --label $label 
#!/bin/bash
category="vbf"
label="Jan07_test"

# python dnn_preprocessor.py --label $label  -cat $category
python dnn_train.py --label $label 
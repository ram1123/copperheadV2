#!/bin/bash

# Note on the out index:
# out_index="0" # sumExp
# out_index="1" # BWZRedux
# out_index="2" # FEWZxBern

# individual fit function ----------------------
for in_index in {0..7}; do # function candidate has 8 in_index and each are frozen for toy generation
    for out_index in {1..1}; do # set core pdf index to BWZ redux, but it is NOT frozen
        sbatch slurm_setup.sub $in_index $out_index
    done
done
# individual fit function ----------------------

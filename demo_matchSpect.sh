#!/bin/bash
# This performs a spectrum design to match the desired characteristics. Note that this may need to be tuned to fit a particular spectrum. It generates random initial conditions on each run.

echo "python scatter_net.py --lr_rate .001 --lr_decay .7 --matchSpectrum True --match_test_file 'results/2_layer_tio2/test_47.5_45.3'"

python scatter_net.py --lr_rate .001 --lr_decay .7 --matchSpectrum True --match_test_file 'results/2_layer_tio2/test_47.5_45.3'
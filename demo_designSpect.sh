#!/bin/bash
# This performs a spectrum design to match the desired characteristics. Note that this may need to be tuned to fit a particular spectrum. It generates random initial conditions on each run.

echo "python scatter_net.py --lr_rate .001 --lr_deacy .7 --designSpectrum True --design_test_file 'data/test_gen_spect.csv'"

python scatter_net.py --lr_rate .001 --lr_decay .01 --designSpectrum True --design_test_file 'data/test_gen_spect.csv'
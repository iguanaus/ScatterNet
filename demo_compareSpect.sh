#!/bin/bash
# This uess a pretrained model to compare the desired and normal output. It generates a Figure like 2b.

echo 'python scatter_net.py --data data/8_layer_tio2 --output_folder results/8_layer_tio2 --n_batch 1 --num_layers 4 --n_hidden 250 --percent_val .2 --patience 10 --sample_val True --compare True --spect_to_sample 20 --lr_rate .001 --lr_decay .99'

python scatter_net.py --data data/8_layer_tio2 --output_folder results/8_layer_tio2 --n_batch 1 --num_layers 4 --n_hidden 250 --percent_val .2 --patience 10 --sample_val True --compare True --spect_to_sample 20 --lr_rate .001 --lr_decay .99

echo 'python plotSpects.py --filename results/8_layer_tio2/test_out_file_20.txt'

python plotSpects.py --filename results/8_layer_tio2/test_out_file_20.txt
#!/bin/bash
# This file trains all the models presented here. 

echo "python scatter_net_1_train.py --data data/8_layer_tio2 --output_folder results/8_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 250 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/8_layer_tio2 --output_folder results/8_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 250 --percent_val .2 --patience 10

echo "python scatter_net_1_train.py --data data/7_layer_tio2 --output_folder results/7_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 225 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/7_layer_tio2 --output_folder results/7_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 225 --percent_val .2 --patience 10

echo "python scatter_net_1_train.py --data data/6_layer_tio2 --output_folder results/6_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 225 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/6_layer_tio2 --output_folder results/6_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 225 --percent_val .2 --patience 10

echo "python scatter_net_1_train.py --data data/5_layer_tio2 --output_folder results/5_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 200 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/5_layer_tio2 --output_folder results/5_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 200 --percent_val .2 --patience 10

echo "python scatter_net_1_train.py --data data/4_layer_tio2 --output_folder results/4_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 125 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/4_layer_tio2 --output_folder results/4_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 125 --percent_val .2 --patience 10

echo "python scatter_net_1_train.py --data data/3_layer_tio2 --output_folder results/3_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 100 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/3_layer_tio2 --output_folder results/3_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 100 --percent_val .2 --patience 10

echo "python scatter_net_1_train.py --data data/2_layer_tio2 --output_folder results/2_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 100 --percent_val .2 --patience 10"
python scatter_net_1_train.py --data data/2_layer_tio2 --output_folder results/2_layer_tio2 --n_batch 100 --numEpochs 5000 --lr_rate .0006 --lr_decay .99 --num_layers 4 --n_hidden 100 --percent_val .2 --patience 10
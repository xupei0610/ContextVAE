# Context-Aware Vehicle Trajectory Prediction with Social Attention and Timewise VAEs

## Dependencies
- Pytorch 1.11
- Numpy 1.21

We recommend to install all the requirements through Conda by

    $ conda create --name <env> --file requirements.txt -c pytorch -c conda-forge

## Code Usage

Train a model from scratch

    $ python main.py \
        --train <train_data_dir> --test <test_data_dir> \
        --config <train_config_file> --ckpt <checkpoint_dir>

Evaluate a trained model

    $ python main.py \
        --test <test_data_dir> \
        --config <eval_config_file> --ckpt <checkpoint_dir>


We provide our configure files in `config` folder.
To reproduce the results, please run

    # nuScenes
    $ python main.py \
        --train <data_dir>/train --test <test_data_dir>/val --map_dir <data_dir>/map \
        --config config/nuscenes_train.py --ckpt <checkpoint_dir>
    
    # Lyft
    $ python main.py \
        --train <data_dir>/train/1 --test <test_data_dir>/validate/1 --map_dir <data_dir>/map \
        --config config/lyft_train.py --ckpt <checkpoint_dir> \
        --rank 0 --workers 4
    $ python main.py \
        --train <data_dir>/train/2 --test <test_data_dir>/validate/2 --map_dir <data_dir>/map \
        --config config/lyft_train.py --ckpt <checkpoint_dir> \
        --rank 1 --workers 4 --master_addr <master_addr>
    $ python main.py \
        --train <data_dir>/train/3 --test <test_data_dir>/validate/3 --map_dir <data_dir>/map \
        --config config/lyft_train.py --ckpt <checkpoint_dir> \
        --rank 2 --workers 4 --master_addr <master_addr>
    $ python main.py \
        --train <data_dir>/train/4 --test <test_data_dir>/validate/4 --map_dir <data_dir>/map \
        --config config/lyft_train.py --ckpt <checkpoint_dir> \
        --rank 3 --workers 4 --master_addr <master_addr>
    
    # Waymo
    $ python main.py \
        --train <data_dir>/training/1 --map_dir <data_dir>/map/training/1 \
        --test <test_data_dir>/validation/1  --test_map_dir <data_dir>/map/validation/1\
        --config config/waymo_train.py --ckpt <checkpoint_dir> \
        --rank 0 --workers 8
    ...
    $ python main.py \
        --train <data_dir>/training/8 --map_dir <data_dir>/map/training/8 \
        --test <test_data_dir>/validation/8  --test_map_dir <data_dir>/map/validation/8\
        --config config/waymo_train.py --ckpt <checkpoint_dir> \
        --rank 7 --workers 8 --master_addr <master_addr>

We use distributed training for `Lyft` and `Waymo` datasets with 4 and 8 worker machines respectively. 
(cf. https://pytorch.org/docs/stable/distributed.html for Pytorch distributed training.)
See [scripts/README.md](scripts) for details of data preprocessing.
All training was done with A100 GPUs for `Lyft` and `Waymo` datasets and a V100 GPU for `nuScenes`.

We also provided our pre-trained models in [Release](https://github.com/xupei0610/ContextVAE/releases/). 
To reproduce the testing results, please run

    # nuScenes
    $ python main.py \
        --test <test_data_dir>/val --map_dir <data_dir>/map \
        --config config/nuscenes_eval.py --ckpt models/nuscenes_res18
    
    # Lyft
    $ python main.py \
        --test <test_data_dir>/validate/1 --map_dir <data_dir>/map \
        --config config/lyft_eval.py --ckpt models/lyft_res152 \
        --rank 0 --workers 4
    $ python main.py \
        --test <test_data_dir>/validate/2 --map_dir <data_dir>/map \
        --config config/lyft_eval.py --ckpt models/lyft_res152 \
        --rank 1 --workers 4 --master_addr <master_addr>
    $ python main.py \
        --test <test_data_dir>/validate/3 --map_dir <data_dir>/map \
        --config config/lyft_eval.py --ckpt models/lyft_res152 \
        --rank 2 --workers 4 --master_addr <master_addr>
    $ python main.py \
        --test <test_data_dir>/validate/4 --map_dir <data_dir>/map \
        --config config/lyft_eval.py --ckpt models/lyft_res152 \
        --rank 3 --workers 4 --master_addr <master_addr>

    # Waymo
    $ python main.py \
        --test <test_data_dir>/validation/1 --map_dir <data_dir>/map/validation/1 \
        --config config/waymo_eval.py --ckpt models/waymo_m2 \
        --rank 0 --workers 8
    ...
    $ python main.py \
        --test <test_data_dir>/validation/8 --map_dir <data_dir>/map/validation/8 \
        --config config/waymo_eval.py --ckpt models/waymo_m2 \
        --rank 7 --workers 8 --master_addr <master_addr>


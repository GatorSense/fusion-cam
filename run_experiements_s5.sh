#!/bin/shd

python3 ./main.py --measure_type 'binary' --experiment_name 's1_gradcam_top_10_choose_4_trn_50' --num_runs 1 --num_sources 4 --initial_num_sources 10 --source_indices 1 2 3 4 5 6 7 8 9 10 --n_choose_k 'y' --bfm_and_ffm 'n' --p 10 -100 --layers 4 --nBagsTrain 50 --nBagsTest 70 --onlyTarget 'y' --nPntsBags 1000 --nActivations 20 --INCLUDE_INVERTED 'n' --SELECTION_APPROACH 'iou'

python3 ./main.py --measure_type 'binary' --experiment_name 's5_gradcam_top_10_choose_4_trn_50' --num_runs 1 --num_sources 4 --initial_num_sources 10 --source_indices 1 2 3 4 5 6 7 8 9 10 --n_choose_k 'y' --bfm_and_ffm 'n' --p 10 -100 --layers 30 --nBagsTrain 50 --nBagsTest 70 --onlyTarget 'y' --nPntsBags 1000 --nActivations 20 --INCLUDE_INVERTED 'n' --SELECTION_APPROACH 'iou'



















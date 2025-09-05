## Step by step instruction
Adaptive segmentation for Steered Mixture of Experts contains four steps:

## Segmentation
Please run matlab code 'test.m' in MDBSCAN-old file
The resulting segmentation will be stored under file '../data/seg/mdbscan/$thres/$partition', where $thres is the threshold setting for segmentation algorithm, and $partition is the name of dataset
## Train SMoE on individual segmentation
Please run the following code:
python data_generation.py 

Here are parameter settings:
--partition,--p: dataset name
--diff: threshold setting in segmentation algorithm

The result are the fixed sized image patches. Each of patches contains each individual segment.
These fixed sized image patches are fed into next step for training multiple SMoE models.

Once the data file is generated, please run:
python Final_train_seg.py
The resulting SMoE models are stored and then we can aggregate the parameters of each SMoE model to formulate our proposed initialization

## Global Exploration
Please run
python load_init_para.py
This will give you the aggregated initial parameters, which is ready for global optimization

## Global Training
Please run
python train_with_init_seperate_grid.py --steer --init_para_path init_para/ --data_path data/img/$partition/ --map_path data/seg/ --partition $partition --result_path seperate_grid --data_name cameraman

--steer: use steered kernel or not
--init_para_path: where the aggregated initial parameters are stored, change it to the output file in previous step
--data_path: where the data is stored
--data_name: the name of the data
--map_path: where the segmentation map stored, default is 'data/seg/'
--result_path: output folder


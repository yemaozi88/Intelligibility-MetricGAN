import os

import numpy as np

repos_dir = r'/home/akikun/repos'
iMetricGAN_dir = os.path.dirname(__file__)

data_dir  = r'/home/common/db/audio_corpora/nele/imgan/all/dataset_5000-500'
train_dir = os.path.join(data_dir, 'train')
Train_Noise_path = os.path.join(train_dir, 'noise')
Train_Clean_path = os.path.join(train_dir, 'clean')
Train_Enhan_path = os.path.join(train_dir, 'enhanced')

test_dir = os.path.join(data_dir, 'test')
Test_Noise_path = os.path.join(test_dir, 'noise')
Test_Clean_path = os.path.join(test_dir, 'clean')

# the directory where the experimental results will be saved.
main_dir    = r'/home/akikun/experiments/jr'
output_path = os.path.join(main_dir, 'output')
pt_dir      = os.path.join(main_dir, 'checkpoint')
log_dir     = os.path.join(main_dir, 'log')

# 1st: SIIB 2nd: ESTOI 
# It can be either 'SIIB' or 'ESTOI' or both for now. 
# Of course, it can be any arbitary metric of interest.
#TargetMetric = 'siib&estoi'
TargetMetric = 'siib'
# Target metric scores you want generator to generate. 
#target_score = np.asarray([1.0, 1.0]) 
target_score = np.asarray([1.0]) 

# number of samples to be used for generator training. 
num_of_sampling = 500
# number of samples to be used for discriminator validation.
num_of_valid_sample = 500

sampling_frequency = 44100
shuffle_dataloader = True
shuffle_paths = True

# training parameters.
GAN_epoch = 300
batch_size = 1
learning_rate_generator = 1e-4
learning_rate_discriminator = 1e-4
num_workers = 6
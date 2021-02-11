import os

import numpy as np

repos_dir = r'/home/akikun/repos'
iMetricGAN_dir = os.path.dirname(__file__)

data_dir = r'/home/common/db/audio_corpora/nele/imgan'
Train_Noise_path = os.path.join(data_dir, 'train_small', 'noise-8')
Train_Clean_path = os.path.join(data_dir, 'train_small', 'clean')
Train_Enhan_path = os.path.join(data_dir, 'train_small', 'enhanced')

Test_Noise_path = os.path.join(data_dir, 'test_small', 'noise-8')
Test_Clean_path = os.path.join(data_dir, 'test_small', 'clean')

# the directory where the experimental results will be saved.
main_dir = r'/home/akikun/experiments/jr'
output_path = os.path.join(main_dir, 'output')
pt_dir = os.path.join(main_dir, 'checkpoint')

# 1st: SIIB 2nd: ESTOI 
# It can be either 'SIIB' or 'ESTOI' or both for now. 
# Of course, it can be any arbitary metric of interest.
TargetMetric = 'siib&estoi'
# Target metric scores you want generator to generate. 
Target_score = np.asarray([1.0,1.0]) 

GAN_epoch = 300
num_of_sampling = 300
num_of_valid_sample = 800
batch_size = 1
sampling_frequency = 44100

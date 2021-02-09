import os

repos_dir = r'/home/akikun/repos'
iMetricGAN_dir = os.path.dirname(__file__)

data_dir = r'/home/common/db/audio_corpora/nele/imgan'
Train_Noise_path = os.path.join(data_dir, 'train_small', 'noise-10')
Train_Clean_path = os.path.join(data_dir, 'train_small', 'clean')
Train_Enhan_path = os.path.join(data_dir, 'train_small', 'enhanced')

Test_Noise_path = os.path.join(data_dir, 'test_small', 'noise-10')
Test_Clean_path = os.path.join(data_dir, 'test_small', 'clean')

main_dir = r'/home/akikun/experiments/jr'
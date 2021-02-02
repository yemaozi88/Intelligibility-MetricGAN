import os

repos_dir = r'/home/akikun/repos'
iMetricGAN_dir = os.path.dirname(__file__)

data_dir = r'/home/common/db/audio_corpora/nele/imetricgan'
Train_Noise_path = os.path.join(data_dir, 'train', 'noise')
Train_Clean_path = os.path.join(data_dir, 'train', 'clean')
Train_Enhan_path = os.path.join(data_dir, 'train', 'enhanced')

Test_Noise_path = os.path.join(data_dir, 'test', 'noise')
Test_Clean_path = os.path.join(data_dir, 'test', 'clean')

main_dir = r'/home/akikun/experiments/jr'
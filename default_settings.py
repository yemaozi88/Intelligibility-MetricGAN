import os

repos_dir = r'/home/akikun/repos'
iMetricGAN_dir = os.path.dirname(__file__)

main_dir = r'/home/akikun/experiments/jr'
Train_Noise_path = os.path.join(main_dir, 'train', 'noise')
Train_Clean_path = os.path.join(main_dir, 'train', 'clean')
Train_Enhan_path = os.path.join(main_dir, 'train', 'enhanced')

Test_Noise_path = os.path.join(main_dir, 'test', 'noise')
Test_Clean_path = os.path.join(main_dir, 'test', 'clean')

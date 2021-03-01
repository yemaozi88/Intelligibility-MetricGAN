# coding=utf-8

import torch
import torch.nn as nn
from audio_util import *
from model import Generator, Discriminator
from dataloader import *
from tqdm import tqdm
from torch.utils.data import *
import os
import librosa
import numpy as np

import default_settings as default


##### MODIFY FROM HERE #####
#chkpt_path = './trained_model/chkpt_26.pt' # location of trained model
chkpt_path = r'/home/akikun/projects/nele/experiments/imgan_siib/checkpoint/chkpt_9.pt'
root_dir = r'/home/common/db/audio_corpora/nele/jr_demo'
Test_Noise_path = os.path.join(root_dir, 'noise')
Test_Clean_path = os.path.join(root_dir, 'clean')
enhanced_dir = os.path.join(root_dir, 'enhanced')
##### MODIFY TILL HERE #####


#batch_size = default.batch_size
#fs = 44100
fs = default.sampling_frequency

print('Reading path of processing data...')
Generator_Test_paths = get_filepaths(Test_Clean_path)
#print(Generator_Test_paths)

# locations of enhanced output files 
if not os.path.exists(enhanced_dir):
    os.makedirs(enhanced_dir)
# mod_path = './inference/English/Modify'
# raw_path = './inference/English/Clean'
# noi_path = './inference/English/Noise'

# os.makedirs(mod_path, exist_ok=True)
# os.makedirs(raw_path, exist_ok=True)
# os.makedirs(noi_path, exist_ok=True)

print('Load Model...')
G = Generator().cuda()
G.load_state_dict(torch.load(chkpt_path)['enhance-model'])

print("Processing...")
G.eval()
with torch.no_grad():
    for path in tqdm(Generator_Test_paths):
        #S = path.split('/')
        #wave_name = S[-1]
        wave_name = os.path.basename(path)
        print(wave_name)

        clean_wav,_ = librosa.load(path, sr=fs)
        noise_wav,_ = librosa.load(os.path.join(Test_Noise_path, wave_name), sr=fs)

        noise_mag,noise_phase = Sp_and_phase(noise_wav, Normalization=True)
        clean_mag,clean_phase = Sp_and_phase(clean_wav, Normalization=True)
        clean_in = clean_mag.reshape(1,clean_mag.shape[0],-1)
        clean_in = torch.from_numpy(clean_in).cuda()
        noise_in = noise_mag.reshape(1,noise_mag.shape[0],-1)
        noise_in = torch.from_numpy(noise_in).cuda()

        mask = G(clean_in, noise_in)
        clean_power = torch.pow(clean_in, 2/0.30)
        beta_2 = torch.sum(clean_power) / torch.sum(torch.pow(mask,2)*clean_power)
        beta_p = beta_2 ** (0.30/2)
        mask = torch.pow(mask, 0.30) * beta_p
        ## Do not change high frequency components in inference stage, since they do not affect intelligibility actually
        #mask[0,:,380:] = 1.0

        enh_mag = clean_in * mask
        enh_mag = (enh_mag**(1/0.30)).detach().cpu().squeeze(0).numpy()
        enh_wav = SP_to_wav(enh_mag.T, clean_phase)

        enh_wav = np.hstack((enh_wav,np.zeros(len(clean_wav)-len(enh_wav),dtype=np.float32)))
        enh_wav = enh_wav / np.std(enh_wav) * np.std(clean_wav)
        
        librosa.output.write_wav(os.path.join(enhanced_dir, wave_name), enh_wav, fs)
#         librosa.output.write_wav(os.path.join(raw_path, wave_name), clean_wav, fs)
#         librosa.output.write_wav(os.path.join(noi_path, wave_name), noise_wav, fs)

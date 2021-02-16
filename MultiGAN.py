# coding=utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess

import torch
import torch.nn as nn
from pystoi.stoi import stoi
from tqdm import tqdm
import pdb

from model import Generator, Discriminator
from audio_util import *
from dataloader import *

import default_settings as default
from torch.utils.tensorboard import SummaryWriter

random.seed(666)

#####  
## load the settings. 
## [TODO] this part should be replaced with hparam.
#####
Train_Noise_path = default.Train_Noise_path
Train_Clean_path = default.Train_Clean_path
Train_Enhan_path = default.Train_Enhan_path

Test_Noise_path = default.Test_Noise_path
Test_Clean_path = default.Test_Clean_path

TargetMetric = default.TargetMetric
Target_score = default.Target_score

output_path = default.output_path
pt_dir = default.pt_dir
log_dir = default.log_dir

GAN_epoch = default.GAN_epoch
num_of_sampling = default.num_of_sampling
num_of_valid_sample = default.num_of_valid_sample
batch_size = default.batch_size
fs = default.sampling_frequency


creatdir(pt_dir)
creatdir(output_path)
creatdir(log_dir)

tensor_log_dir = os.path.join(log_dir, 'logs')
tensor_writer = SummaryWriter(log_dir=tensor_log_dir)
#########################  Training data #######################
print('Reading path of training data...')
Generator_Train_paths = get_filepaths(Train_Clean_path)
random.shuffle(Generator_Train_paths)
######################### validation data #########################
print('Reading path of validation data...')
Generator_Test_paths = get_filepaths(Test_Clean_path) 
random.shuffle(Generator_Test_paths)
################################################################
G = Generator().cuda()

D = Discriminator().cuda()
MSELoss = nn.MSELoss().cuda()

optimizer_g = torch.optim.Adam(G.parameters(), lr=1e-3)
optimizer_d = torch.optim.Adam(D.parameters(), lr=1e-3)

Test_STOI = []
Test_SIIB = []

Previous_Discriminator_training_list = []
# [Aki] Why?? output_path is just made above.
shutil.rmtree(output_path)

step_g = 0
step_d = 0

for gan_epoch in np.arange(1, GAN_epoch+1):

    # Prepare directories
    #creatdir(output_path+"/epoch"+str(gan_epoch))
    creatdir(os.path.join(output_path, "epoch" + str(gan_epoch)))
    #creatdir(output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch))
    creatdir(os.path.join(
        output_path, 
        "epoch" + str(gan_epoch), 
        "Test_epoch" + str(gan_epoch)))
    #creatdir(output_path+'/For_discriminator_training')
    creatdir(os.path.join(output_path, 'For_discriminator_training'))
    #creatdir(output_path+'/temp')
    creatdir(os.path.join(output_path, 'temp'))

    # random sample some training data 
    # randomize every epoch.
    random.shuffle(Generator_Train_paths)
    genloader = create_dataloader(Generator_Train_paths[0:round(1*num_of_sampling)],Train_Noise_path)

    if gan_epoch>=2:
        print('Generator training (with discriminator fixed)...')
        for clean_mag,clean_phase,noise_mag,noise_phase, target in tqdm(genloader):
            clean_mag = clean_mag.cuda()
            noise_mag = noise_mag.cuda()
            target = target.cuda()

            mask = G(clean_mag, noise_mag)

            clean_power = torch.pow(clean_mag.detach(), 2/0.30)
            beta_2 = torch.sum(clean_power) / torch.sum(torch.pow(mask,2)*clean_power)
            beta_p = beta_2 ** (0.30/2)
            beta = beta_2 ** 0.5

            enh_mag = clean_mag * torch.pow(mask, 0.30) * beta_p
            ref_mag = clean_mag.detach()

            enh_mag = enh_mag.view(1,1,enh_mag.shape[1],enh_mag.shape[2]).transpose(2,3).contiguous()
            noise_mag = noise_mag.view(1,1,noise_mag.shape[1],noise_mag.shape[2]).transpose(2,3).contiguous()
            ref_mag = ref_mag.view(1,1,ref_mag.shape[1],ref_mag.shape[2]).transpose(2,3).contiguous()
            d_inputs = torch.cat((enh_mag,noise_mag,ref_mag),dim=1)

            score = D(d_inputs)

            loss = MSELoss(score, target)
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            step_g += 1

            if step_g % 200 ==0:
                print((beta*mask).max())
                print((beta*mask).min())
                print('Step %d: Loss in G training is %.3f'%(step_g,loss.item()))

    # Evaluate the performance of generator in a validation set.
    interval_epoch = 1
    if gan_epoch % interval_epoch == 0: 
        print('Evaluate G by validation data ...')
        Test_enhanced_Name = []
        utterance = 0
        G.eval()
        with torch.no_grad():
            for i, path in enumerate(Generator_Test_paths[0:num_of_valid_sample]):
                S = path.split('/')
                wave_name = S[-1]
                #print(wave_name)

                clean_wav,_ = librosa.load(path, sr=fs)
                noise_wav,_ = librosa.load(os.path.join(Test_Noise_path, wave_name), sr=fs)
                noise_mag,noise_phase = Sp_and_phase(noise_wav, Normalization=True)
                clean_mag,clean_phase = Sp_and_phase(clean_wav, Normalization=True)
                
                clean_in = clean_mag.reshape(1,clean_mag.shape[0],-1)
                clean_in = torch.from_numpy(clean_in).cuda()
                noise_in = noise_mag.reshape(1,noise_mag.shape[0],-1)
                noise_in = torch.from_numpy(noise_in).cuda()
                # Energy normalization
                mask = G(clean_in, noise_in)
                clean_power = torch.pow(clean_in, 2/0.30)
                beta_2 = torch.sum(clean_power) / torch.sum(torch.pow(mask,2)*clean_power)
                beta_p = beta_2 ** (0.30/2)
                enh_mag = clean_in * torch.pow(mask, 0.30) * beta_p
                enh_mag = (enh_mag**(1/0.30)).detach().cpu().squeeze(0).numpy()
                enh_wav = SP_to_wav(enh_mag.T, clean_phase)

                #wav_id = wave_name.replace('.wav', '')
                if utterance<20: # Only seperatly save the firt 20 utterance for listening comparision 
                    #enhanced_name=output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch)+"/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
                    enhanced_name = os.path.join(
                        output_path, 
                        "epoch" + str(gan_epoch), 
                        "Test_epoch" + str(gan_epoch), 
                        wave_name)
                else:
                    #enhanced_name=output_path+"/temp"+"/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
                    enhanced_name = os.path.join(
                        output_path, 
                        "temp",
                        wave_name)
            
                librosa.output.write_wav(enhanced_name, enh_wav, fs)
                for t in np.arange(0, len(enh_wav)):
                    tensor_writer.add_scalar(wave_name, enh_wav[t], t)
                utterance+=1      
                Test_enhanced_Name.append(enhanced_name) 
                #print(i)
        G.train()

        # Calculate True STOI
        test_STOI = read_batch_STOI(Test_Clean_path, Test_Noise_path, Test_enhanced_Name)
        Test_STOI.append(np.mean(test_STOI))
        # Calculate True SIIB
        test_SIIB = read_batch_SIIB(Test_Clean_path, Test_Noise_path, Test_enhanced_Name)
        Test_SIIB.append(np.mean(test_SIIB))
        with open(os.path.join(log_dir, 'log.txt'), 'a') as f:
            f.write('SIIB is %.3f, ESTOI is %.3f\n'%(np.mean(test_SIIB), np.mean(test_STOI)))

        # Plot learning curves
        plt.figure(1)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_STOI,'b',label='ValidSTOI')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('ESTOI')
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(log_dir, 'ESTOI.png'), dpi=150)
        
        plt.figure(2)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_SIIB,'r',label='ValidSIIB')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('SIIB')
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(log_dir, 'SIIB.png'), dpi=150)
    
    # save the current enhancement model
    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % gan_epoch)
    torch.save({
        'enhance-model': G.state_dict(),
    }, save_path)

    print('Sample training data for discriminator training...')
    D_paths = Generator_Train_paths[0:num_of_sampling]

    Enhanced_name = []

    G.eval()
    with torch.no_grad():
        for path in D_paths:
            S = path.split('/')
            wave_name = S[-1]
            clean_wav,_ = librosa.load(path, sr=fs)
            #noise_wav,_ = librosa.load(Train_Noise_path+wave_name, sr=fs)
            noise_wav,_ = librosa.load(os.path.join(Train_Noise_path, wave_name), sr=fs)
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
            enh_mag = clean_in * torch.pow(mask, 0.30) * beta_p
            enh_mag = (enh_mag**(1/0.30)).detach().cpu().squeeze(0).numpy()
            enh_wav = SP_to_wav(enh_mag.T, clean_phase)

            wav_id = wave_name.replace('.wav', '')
            #enhanced_name=output_path+"/For_discriminator_training/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
            enhanced_name = os.path.join(
                output_path, 
                "For_discriminator_training", 
                wave_name)

            librosa.output.write_wav(enhanced_name, enh_wav, fs)
            Enhanced_name.append(enhanced_name)

    G.train()

    if TargetMetric=='siib&estoi':
        # Calculate True SIIB score
        train_SIIB = read_batch_SIIB(Train_Clean_path, Train_Noise_path, Enhanced_name)
        train_STOI = read_batch_STOI(Train_Clean_path, Train_Noise_path, Enhanced_name)
        train_SIIB = List_concat_score(train_SIIB, train_STOI)
        current_sampling_list=List_concat(train_SIIB, Enhanced_name) # This list is used to train discriminator.
    
        #DRC_Enhanced_name = [Train_Enhan_path+'Train_'+S.split('/')[-1].split('_')[-1].split('@')[0]+'.wav' for S in Enhanced_name]        
        DRC_Enhanced_name = [os.path.join(Train_Enhan_path, os.path.basename(S)) for S in Enhanced_name]
        #pdb.set_trace()
        train_SIIB_DRC = read_batch_SIIB_DRC(Train_Clean_path, Train_Noise_path, DRC_Enhanced_name)
        train_STOI_DRC = read_batch_STOI_DRC(Train_Clean_path, Train_Noise_path, DRC_Enhanced_name)
        train_SIIB_DRC = List_concat_score(train_SIIB_DRC, train_STOI_DRC)
        Co_DRC_list = List_concat(train_SIIB_DRC, DRC_Enhanced_name)

    print("Discriminator training...")
    # Training for current list
    Current_Discriminator_training_list = current_sampling_list+Co_DRC_list
    #print(Current_Discriminator_training_list)
    #pdb.set_trace()

    random.shuffle(Current_Discriminator_training_list)

    disloader = create_dataloader(Current_Discriminator_training_list, Train_Noise_path, Train_Clean_path, loader='D')

    for x,target in tqdm(disloader):
        x = x.cuda()
        target = target.cuda()
        score = D(x)
        loss = MSELoss(score, target)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()
        step_d += 1
        #if step_d % 1000 ==0:
        #    print('Step %d: Loss in D training is %.3f'%(step_d,loss.item()))
    

    ## Training for current list + Previous list (like replay buffer in RL, optional)
    random.shuffle(Previous_Discriminator_training_list)

    Total_Discriminator_training_list=Previous_Discriminator_training_list[0:len(Previous_Discriminator_training_list)//25]+Current_Discriminator_training_list # Discriminator_Train_list is the list used for pretraining.
    random.shuffle(Total_Discriminator_training_list)

    disloader_past = create_dataloader(Total_Discriminator_training_list, Train_Noise_path, Train_Clean_path, loader='D')

    for x,target in tqdm(disloader_past):
        x = x.cuda()
        target = target.cuda()
        score = D(x)
        loss = MSELoss(score, target)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()
        step_d += 1
        #if step_d % 1000 ==0:
        #    print('Step %d: Loss in D training is %.3f'%(step_d,loss.item()))
        
    # Update the history list
    Previous_Discriminator_training_list=Previous_Discriminator_training_list+Current_Discriminator_training_list 
    
    # Training current list again
    for x,target in tqdm(disloader):
        x = x.cuda()
        target = target.cuda()
        score = D(x)
        loss = MSELoss(score, target)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()
        step_d += 1
        #if step_d % 1000 ==0:
        #    print('Step %d: Loss in D training is %.3f'%(step_d,loss.item()))
    
    #shutil.rmtree(output_path+'/temp')
    shutil.rmtree(os.path.join(output_path, 'temp'))
    
    print('Epoch %d Finished' % gan_epoch)

tensor_writer.close()
print('Finished')

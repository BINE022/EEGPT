from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
import tqdm
import pandas as pd
import random
import tqdm
import os,sys
from models.modules.electrode import electrode_names, \
    electrode_name2idx, electrode_num, raw_chs2chan_ids, electrode_none_idx

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)


class Fusion_eeg_dataset(Dataset):
    def __init__(self,data_info_1, data_info_2, root_dir_1, root_dir_2,
                 percentile_q = 0,
                 use_tanh  = False):
        """
        Args:
            data_info (DataFram): file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.percentile_q = percentile_q
        self.use_tanh  = use_tanh
        self.electrode_names = electrode_names
        
        self.data_info_1 = data_info_1
        self.data_info_2 = data_info_2
        self.root_dir_1  = root_dir_1
        self.root_dir_2  = root_dir_2

        self.cache_chs   = {}
        self.cache_x_type= {}
        self.cache_norm  = {}
         
        self.t = 5
        self.freq = 250
        self.durition = int(self.t*self.freq)
        self.part_1_len = len(self.data_info_1)

    def padding_x(self,raw_x,raw_chs,durition):

        '''
        Padding the raw_x to have same shape in spatial dimension.
        raw_x : the orginal x.
        begin : start positon in temporal dimension.
        t     : The duration is t seconds.
        '''
        raw_chs = [i.upper() for i in raw_chs]

        x = torch.zeros(electrode_num,durition)

        # 标记数据中是否为EEG信号
        x_type = torch.zeros(durition)
        x_type[:raw_x.shape[1]] = 1 

        for i,ch_name in enumerate(raw_chs):
            index = electrode_name2idx[ch_name]
            x[index,:raw_x.shape[1]] = raw_x[i]
    
        chs = raw_chs2chan_ids(raw_chs)

        return x,chs,x_type


    def set_t(self,t):
        self.t = t
        self.durition = int(self.t*self.freq)
  

    def __len__(self):
        return len(self.data_info_1) + len(self.data_info_2)

    def __getitem__(self, idx):

        
        if idx < self.part_1_len:

            s_info = self.data_info_1.iloc[idx]

            # 短于5min的数据进行重新抽取
            if s_info['trials_len'] < 5*60*250:
                idx = random.randint(0,len(self.data_info_1)-1)
                return self.__getitem__(idx)
            
            # 从数据中截取1min和-1min的数据，并进行随机抽取
            begin = random.randint(60*250,s_info['trials_len']-60*250)

            split_len = 1000
            begin_index = begin - (begin//split_len)*split_len
            begin_num = begin//split_len
            
            all_parts = []
            total_len = 0
            i = 0
            while total_len < self.durition:
                sample_path = os.path.join(self.root_dir_1,s_info['data_path'],'{}_part_{}.npy'.format(s_info['ids'],i + begin_num))
                data = np.load(sample_path)
                if i == 0:
                    part_data = torch.FloatTensor(data[:,begin_index:])
                else:
                    part_data = torch.FloatTensor(data)
                i+=1
                total_len += part_data.shape[1]
                all_parts.append(part_data)

            raw_x = torch.cat(all_parts,dim = 1)[:,:self.durition]
            raw_chs = s_info['channels']
            # raw_chs = eval(s_info['channels'])

        else:
            sample_info = self.data_info_2.iloc[idx-self.part_1_len]
            raw_x       = torch.FloatTensor(np.load(os.path.join(self.root_dir_2,sample_info['data_path'])))
            end_index   = raw_x.shape[1]-self.durition
            # end_index   = end_index if end_index > 0 else raw_x.shape[1]//2
            if end_index > 0:
                random_began= random.randint(0, end_index)
            else:
                random_began= 0
            raw_x   = raw_x[:,random_began:(random_began+self.durition)]
            raw_chs = sample_info['channels']
        
        raw_time_len = raw_x.shape[1]
        tmp = raw_time_len%self.freq
        if tmp>0:
            raw_x = raw_x[:,:(-tmp)]
            raw_time_len -= tmp
        
        if raw_chs in self.cache_chs:
            chs, chs_map = self.cache_chs[raw_chs]
        else:
            key     = raw_chs
            raw_chs = [i.upper() for i in eval(raw_chs)]
            chs     = raw_chs2chan_ids(raw_chs)
        
            chs_map = ([],[])
            for i,ch_name in enumerate(raw_chs):
                index = electrode_name2idx[ch_name]
                chs_map[0].append(index)
                chs_map[1].append(i)
            chs_map = (torch.tensor(chs_map[0], dtype=torch.long),
                       torch.tensor(chs_map[1], dtype=torch.long))
            self.cache_chs[key] = (chs, chs_map)
            
        if raw_time_len in self.cache_x_type:
            x_type = self.cache_x_type[raw_time_len]
        else:
            # 标记数据中是否为EEG信号
            x_type = torch.zeros(self.durition)
            x_type[:raw_time_len] = 1 
            self.cache_chs[raw_time_len] = x_type
            
        x = torch.zeros(electrode_num,self.durition)
        
        
        if self.percentile_q>0:
            if idx in self.cache_norm:
                norm = self.cache_norm[idx]
            else:
                norm = torch.quantile(raw_x.abs(), self.percentile_q)
                self.cache_norm[idx] = norm
                
            raw_x = raw_x / (norm + 1e-6)
            if self.use_tanh:
                raw_x = torch.tanh(raw_x)
                
        x[chs_map[0],:raw_time_len] = raw_x[chs_map[1]]
        
        return x,chs,x_type




class Fake_eeg_dataset(Dataset):
    def __init__(self,):
        
        self.t = 5
        self.durition= int(self.t*250)
        self.freq = 250

    def set_t(self,t):
        self.t = t
        self.durition= int(self.t*250)
  
    def __len__(self):
        return 1024

    def __getitem__(self, idx):

        x       = torch.randn((electrode_num, self.durition))
        chs     = torch.arange(electrode_num).long()
        x_type  = torch.ones(self.durition)
        return x,chs,x_type


def get_data(T = 5,
             percentile_q = 0, 
             use_tanh=False,
             use_tiny_data=False,
             use_fake_data=False,
             ):
    
    if use_fake_data:
        train_dataset = Fake_eeg_dataset()
        valid_dataset = Fake_eeg_dataset()
        test_dataset  = Fake_eeg_dataset()

        train_dataset.set_t(T)
        valid_dataset.set_t(T)
        test_dataset.set_t(T)
    else:
        
        # data_dir_1 = r'/disks/SSD2/dataset/TUHEEG_Processed'
        data_dir_1 = r'/disks/HDD3/dataset/TUHEEG_Processed'
        data_info_path_1 = os.path.join(data_dir_1, 'all_samples_info.csv')
        info_1 = pd.read_csv(data_info_path_1)

        data_dir_2 = os.path.join(r'/disks/SSD/data/transformed2')
        data_info_path_2 = os.path.join(r'/disks/HDD3/dataset/Data_processed/Data_processed/all_samples_info.csv')
        info_2 = pd.read_csv(data_info_path_2)
        
        train_info_1 = info_1[(info_1['ids']<69000)]
        valid_info_1 = info_1[(info_1['ids']>=69000) & (info_1['ids']<69002)]
        test_info_1  = info_1[(info_1['ids']>=69032)]

        # train_info_2   =  info_2[(info_2['Dataset']!='Zhou2016')&(info_2['Dataset']!='BNCI2015_001')&(info_2['Dataset']!='BNCI2014_001')&(info_2['Dataset']!='AlexMI')&(info_2['Dataset']!='BNCI2014_002')&(info_2['Dataset']!='BNCI2014_004')]
        # train_info_2   =  info_2[(info_2['Dataset']=='Stieger2021')|(info_2['Dataset']=='Schirrmeister2017')]
        # train_info_2   =  info_2[(info_2['Dataset'] == 'Stieger2021')|(info_2['Dataset']=='Schirrmeister2017')]

        train_info_2     =  info_2[(info_2['Dataset']!='Cho2017')&(info_2['Dataset']!='PhysionetMI')&(info_2['Dataset']!='Zhou2016')&(info_2['Dataset']!='BNCI2015_001')&(info_2['Dataset']!='BNCI2014_001')&(info_2['Dataset']!='AlexMI')&(info_2['Dataset']!='BNCI2014_002')&(info_2['Dataset']!='BNCI2014_004')]
        valid_info_2     =  info_2[(info_2['Dataset']=='Zhou2016')|(info_2['Dataset']=='BNCI2015_001')|(info_2['Dataset']=='AlexMI')]
        test_info_2      =  info_2[(info_2['Dataset']=='BNCI2014_001')]
        
        if use_tiny_data:
            train_info_1 = train_info_1.iloc[:len(valid_info_1)]
            train_info_2 = train_info_2.iloc[:len(valid_info_2)]

        train_dataset = Fusion_eeg_dataset(train_info_1,train_info_2,data_dir_1,data_dir_2, percentile_q, use_tanh)
        valid_dataset = Fusion_eeg_dataset(valid_info_1,valid_info_2,data_dir_1,data_dir_2, percentile_q, use_tanh)
        test_dataset  = Fusion_eeg_dataset(test_info_1, test_info_2, data_dir_1,data_dir_2, percentile_q, use_tanh)

        train_dataset.set_t(T)
        valid_dataset.set_t(T)
        test_dataset.set_t(T)
        
    return train_dataset,valid_dataset,test_dataset


def main():
    train_dataset,valid_dataset,test_dataset = get_data()
    
    train_loader = DataLoader(train_dataset,num_workers=0,  pin_memory=True,   batch_size = 1024,shuffle = True,  drop_last=True)
    # valid_loader = DataLoader(valid_dataset,num_workers=10,  batch_size = 200,  shuffle = False, drop_last=False)
    # test_loader  = DataLoader(test_dataset, num_workers=10,  batch_size = 200,  shuffle = False, drop_last=False)
    from models.modules.electrode import electrode_none_idx
    
    min_chan_num = 10000
    for i,(x,chs,x_type) in tqdm.tqdm(enumerate(train_loader),total = len(train_loader)):
        # print(x.shape, chs.shape, x_type.shape)
        print(x.max(),x.min(),x.mean(),x.std())
        n = torch.sum(chs!=electrode_none_idx, dim=-1).min().item()
        min_chan_num = min(min_chan_num,n)
        break
    print(min_chan_num)
    # for data in valid_loader:
    #     pass
    # for data in test_loader:
    #     pass



    
        
    

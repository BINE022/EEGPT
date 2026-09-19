"""
use this script to prepare the mixed pretraining dataset

Requires torcheeg==1.1.0 (dataset classes and their chunk_size defaults vary across
versions) and mne==1.4.2. Builds the PhysioNetMI / tsu_benchmark / seed / m3cv / HGD
tags into ./merged/{Train,Valid}Folder/0/.
"""

import os
import torch
import shutil
import random
import mne

# scipy >= 1.13 removed scipy.signal.hann, which torcheeg 1.1.0 imports at module
# load. The PhysioNetMI io build additionally runs joblib/loky workers, which start
# fresh interpreters that do NOT execute this shim -- for that stage put these three
# lines into a sitecustomize.py and prepend its directory to PYTHONPATH.
import scipy.signal as _scipy_signal
import scipy.signal.windows as _scipy_windows
if not hasattr(_scipy_signal, "hann"):
    _scipy_signal.hann = _scipy_windows.hann

import pandas as pd
from torcheeg.datasets import CSVFolderDataset
from torcheeg import transforms
import copy

import torcheeg
import torch
from torcheeg.datasets import M3CVDataset, TSUBenckmarkDataset, DEAPDataset, SEEDDataset, moabb
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LIST, M3CV_CHANNEL_LIST, TSUBENCHMARK_CHANNEL_LIST
from torcheeg.datasets import CSVFolderDataset
from torchaudio.transforms import Resample

# ------------------- PhysioMI
data_root_path = "./io_root/"


PHYSIONETMI_CHANNEL_LIST = ['Fc5.', 'Fc3.', 'Fc1.', 
                            'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 
                            'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 
                            'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 
                            'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 
                            'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 
                            'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 
                            'Oz..', 'O2..', 'Iz..']
PHYSIONETMI_CHANNEL_LIST = [x.strip('.').upper() for x in PHYSIONETMI_CHANNEL_LIST]


use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]

def temporal_interpolation(x, desired_sequence_length, mode='nearest'):
    # squeeze and unsqueeze because these are done before batching
    x = x - x.mean(-2)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
    


def get_physionet_dataset():
    channels_name = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']

    """
    In summary, the experimental runs were:

    1   Baseline, eyes open
    2   Baseline, eyes closed
    3   Task 1 (open and close left or right fist)                  -> 4 5
    4   Task 2 (imagine opening and closing left or right fist)     -> 0 1
    5   Task 3 (open and close both fists or both feet)             -> 6 7
    6   Task 4 (imagine opening and closing both fists or both feet)-> 2 3
    7   Task 1
    8   Task 2
    9   Task 3
    10  Task 4
    11  Task 1
    12  Task 2
    13  Task 3
    14  Task 4

    """
    session_id2task_id = {
        3:1, 4:2, 5:3, 6:4,
        7:1, 8:2, 9:3, 10:4,
        11:1, 12:2, 13:3, 14:4,
        
    }
    task2event_id = {
        0:dict([('T1', 4), ('T2', 5)]),
        1:dict([('T1', 0), ('T2', 1)]),
        2:dict([('T1', 6), ('T2', 7)]),
        3:dict([('T1', 2), ('T2', 3)])
    }

    
    if not os.path.exists(data_root_path+'io/PhysioNetMI'):
        src_path = "./PhysioNetMI/files/eegmmidb/1.0.0/"
        ls = []
        channels_name = None
        for subject in range(1,110):
            for task in [0,1,2,3]:
                for session in [3,7,11]:
                    session += task
                    file_path = src_path + "S{:03d}".format(subject) + '/' + "S{:03d}R{:02d}.edf".format(subject,session)
                    raw = mne.io.read_raw_edf(file_path,preload=True)
                    
                    if channels_name is None:
                        channels_name = copy.deepcopy(raw.ch_names)
                    else:
                        assert channels_name == raw.ch_names
                        
                    event_id = task2event_id[session_id2task_id[session]-1]
                    # -- split epochs
                    epochs = mne.Epochs(raw, 
                            events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)[0], 
                            tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                            preload=True, 
                            decim=1,
                            baseline=None, 
                            reject_by_annotation=False)
                    
                    d = {
                        # "subject_id":[subject],
                        # "sess_id": [session],
                        # "task_id":[task],
                        "file_path":[file_path],
                        "labels":"".join([str(ev[-1]) for ev in epochs.events])
                    }
                    
                    ls.append(pd.DataFrame(d))
        table = pd.concat(ls, ignore_index=True)
        # print(table)
        print(channels_name)
        table.to_csv("./PhysioNetMI/physionetmi_meta.csv", index=False)

        def default_read_fn(file_path, task_id=None, session_id=None, subject_id=None, **kwargs):
            session_id = int(file_path.split('R')[-1].split('.')[0])
            # -- read raw file
            raw = mne.io.read_raw_edf(file_path,preload=True)
            
            event_id = task2event_id[session_id2task_id[session_id]-1]
            # -- split epochs
            epochs = mne.Epochs(raw, 
                    events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)[0], 
                    tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                    preload=True, 
                    decim=1,
                    baseline=None, 
                    reject_by_annotation=False)
            
            return epochs

        dataset = CSVFolderDataset(csv_path="./PhysioNetMI/physionetmi_meta.csv",
                                read_fn=default_read_fn,
                                io_path=data_root_path+'io/PhysioNetMI',
                            online_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.To2d()
                            ]),
                                #    label_transform=transforms.Select('label'),
                                num_worker=4)
    dataset = CSVFolderDataset(
                        io_path=data_root_path+'io/PhysioNetMI',
                        online_transform=transforms.Compose([
                            transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, PHYSIONETMI_CHANNEL_LIST)),
                            transforms.ToTensor(),
                            #   transforms.RandomWindowSlice(window_size=160*4, p=1.0),
                            transforms.Lambda(lambda x: temporal_interpolation(x, 256*6) * 1e3), #V-> 1000uV
                            transforms.To2d()
                        ]),
                        label_transform=transforms.Compose([
                            #   transforms.Select('labels'),
                            #   transforms.StringToInt()
                            transforms.Lambda(lambda x : 0)
                        ]))
    return dataset


# --------------- merge to Fold Dataset

def get_TSU_dataset():
    dataset = TSUBenckmarkDataset(
            root_path="./TSUBenchmark/",
            io_path=data_root_path+'io/tsu_benchmark',
            # 1000 points @ 250 Hz = 4 s
            chunk_size=1000,
            online_transform=transforms.Compose([
                transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, TSUBENCHMARK_CHANNEL_LIST)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: temporal_interpolation(x, 256*4) / 1000),# 1000uV
                transforms.To2d(),
            ]),
            label_transform=transforms.Select('trial_id'))
    return dataset


def get_M3CV_dataset():
    dataset = M3CVDataset(
                        root_path="./aistudio/",
                        io_path=data_root_path+'io/m3cv',
                        online_transform=transforms.Compose([
                            transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, M3CV_CHANNEL_LIST)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: temporal_interpolation(x, 256*4) / 1000),# 1000uV
                            transforms.To2d(),
                        ]),
                        label_transform=transforms.Compose([
                            transforms.Select('subject_id'),
                            transforms.StringToInt()
                        ]))
    return dataset

def get_SEED_dataset():
    dataset = SEEDDataset(
                            # root_path must be the Preprocessed_EEG content layer
                            # (label.mat, readme.txt, *.mat) -- pointing at the parent
                            # folder crashes SEEDDataset on channel-order.xlsx
                            root_path="./SEED/",
                            io_path=data_root_path+'io/seed',
                            # 2000 points @ 200 Hz = 10 s
                            chunk_size=2000,
                          online_transform=transforms.Compose([
                              transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, SEED_CHANNEL_LIST)),
                              transforms.ToTensor(),
                            #   transforms.RandomWindowSlice(window_size=250*4, p=1.0),
                              transforms.Lambda(lambda x: temporal_interpolation(x, 256*10)/1000),# 1000uV
                              transforms.To2d(),                          
                          ]),
                          label_transform=transforms.Compose([
                              transforms.Select('emotion'),
                              transforms.Lambda(lambda x: x + 1)
                          ]))
    return dataset


# --------------- HGD

HGD_CHANNEL_NAMES = ['EEG Fp1', 'EEG Fp2', 'EEG Fpz',
                 'EEG F7', 'EEG F3', 'EEG Fz', 'EEG F4', 'EEG F8',
                 'EEG FC5', 'EEG FC1', 'EEG FC2', 'EEG FC6',
                 'EEG M1', 'EEG T7', 'EEG C3', 'EEG Cz', 'EEG C4', 'EEG T8', 'EEG M2',
                 'EEG CP5', 'EEG CP1', 'EEG CP2', 'EEG CP6',
                 'EEG P7', 'EEG P3', 'EEG Pz', 'EEG P4', 'EEG P8',

                 'EEG POz', 'EEG O1', 'EEG Oz', 'EEG O2',

                 'EOG EOGh', 'EOG EOGv', 'EMG EMG_RH', 'EMG EMG_LH', 'EMG EMG_RF',

                 'EEG AF7', 'EEG AF3', 'EEG AF4', 'EEG AF8',
                 'EEG F5', 'EEG F1', 'EEG F2', 'EEG F6',

                 'EEG FC3', 'EEG FCz', 'EEG FC4',
                 'EEG C5', 'EEG C1', 'EEG C2', 'EEG C6',
                 'EEG CP3', 'EEG CPz', 'EEG CP4',
                 'EEG P5', 'EEG P1', 'EEG P2', 'EEG P6',
                 'EEG PO5', 'EEG PO3', 'EEG PO4', 'EEG PO6',
                 'EEG FT7', 'EEG FT8', 'EEG TP7', 'EEG TP8',
                 'EEG PO7', 'EEG PO8',
                 'EEG FT9', 'EEG FT10',

                 'EEG TPP9h', 'EEG TPP10h',

                 'EEG PO9', 'EEG PO10', 'EEG P9', 'EEG P10', 'EEG AFF1', 'EEG AFz', 'EEG AFF2', 'EEG FFC5h', 'EEG FFC3h', 'EEG FFC4h', 'EEG FFC6h', 'EEG FCC5h',
                 'EEG FCC3h', 'EEG FCC4h', 'EEG FCC6h', 'EEG CCP5h', 'EEG CCP3h', 'EEG CCP4h', 'EEG CCP6h', 'EEG CPP5h', 'EEG CPP3h', 'EEG CPP4h', 'EEG CPP6h',
                 'EEG PPO1', 'EEG PPO2', 'EEG I1', 'EEG Iz', 'EEG I2', 'EEG AFp3h', 'EEG AFp4h', 'EEG AFF5h', 'EEG AFF6h', 'EEG FFT7h', 'EEG FFC1h', 'EEG FFC2h',
                 'EEG FFT8h', 'EEG FTT9h', 'EEG FTT7h', 'EEG FCC1h', 'EEG FCC2h', 'EEG FTT8h', 'EEG FTT10h', 'EEG TTP7h', 'EEG CCP1h', 'EEG CCP2h', 'EEG TTP8h',
                 'EEG TPP7h', 'EEG CPP1h', 'EEG CPP2h', 'EEG TPP8h', 'EEG PPO9h', 'EEG PPO5h', 'EEG PPO6h', 'EEG PPO10h', 'EEG POO9h', 'EEG POO3h', 'EEG POO4h',
                 'EEG POO10h', 'EEG OI1h', 'EEG OI2h']


def get_HGD_dataset():
    # plain mne pipeline (no torcheeg io); deterministic subject -> phase -> epoch
    # order; yields (tensor [58, 2560] float64, sample name)
    mapping = {k: (k[4:]).upper() for k in HGD_CHANNEL_NAMES}
    stims = [1, 2, 3, 4]
    for subject in range(1, 15):
        for phase in ['train', 'test']:
            file_path = os.path.join("./high_gamma/", phase, '{}.edf'.format(subject))
            raw_eeg = mne.io.read_raw_edf(file_path)
            raw_eeg.rename_channels(mapping)
            raw_eeg.pick_channels(use_channels_names)
            events, event_id = mne.events_from_annotations(raw_eeg)
            epoch = mne.Epochs(raw_eeg, events, event_id=stims, tmin=0,
                               tmax=10 - 1 / raw_eeg.info['sfreq'], baseline=None,
                               preload=True, proj=False)
            echans = [x.upper().strip('.') for x in epoch.ch_names]
            choice_channels = [echans.index(ch.upper()) for ch in use_channels_names]
            epoch = epoch.resample(256)      # resample
            x_data = epoch.get_data() * 1000 # unify units (V -> 1000 uV)
            for i in range(len(x_data)):
                data = torch.tensor(x_data[i])[choice_channels] # unify channel order
                data = data - data.mean(-2) # average reference
                yield data.clone().detach(), f"HGD_{phase}_s{subject}_{i}"


if __name__=="__main__":
    import random
    import os
    import tqdm

    for tag in ["PhysioNetMI", "tsu_benchmark", "seed", "m3cv", "HGD"]:
        if tag == "HGD":
            samples = get_HGD_dataset() # yields (tensor, sample name)
        else:
            if tag == "PhysioNetMI":
                dataset = get_physionet_dataset()
            elif tag == "tsu_benchmark":
                dataset = get_TSU_dataset()
            elif tag == "m3cv":
                dataset = get_M3CV_dataset()
            elif tag == "seed":
                dataset = get_SEED_dataset()
            else:
                raise ValueError("Invalid tag")
            print(len(dataset))
            print(dataset[0][0].shape)
            print(dataset[0][0].min(),dataset[0][0].max(), dataset[0][0].mean(),dataset[0][0].std())
            samples = ((x, tag+f"_{i}") for i, (x,y) in enumerate(dataset))
        for x, sample_name in tqdm.tqdm(samples):
            dst="./merged/"
            if random.random()<0.1:
                dst+="ValidFolder/0/"
            else:
                dst+="TrainFolder/0/"
            os.makedirs(dst, exist_ok=True)
            data = x.squeeze_(0)
            # data = data.clone().detach().cpu()
            print(sample_name, data.shape, len(data.shape)==2 and data.shape[0]==58 and data.shape[1]>=1024)
            assert len(data.shape)==2 and data.shape[0]==58 and data.shape[1]>=1024
            torch.save(data, dst + sample_name + ".edf")
            del data, x
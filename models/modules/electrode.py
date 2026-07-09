electrode_names = ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'FP1H', 'FP2H', 'AF1', 'AF2', 'AF5', 'AF6', 'F9', 'F10', 'FT9', 'FT10', 'TP9', 'TP10', 'P9H', 'P10H', 'F1H', 'F2H', 'F5H', 'F6H', 'F7H', 'F8H', 'FC1H', 'FC2H', 'FC5H', 'FC6H', 'FT1', 'FT2', 'C1H', 'C2H', 'C5H', 'C6H', 'T1', 'T2', 'TP1', 'TP2', 'CP1H', 'CP2H', 'CP5H', 'CP6H', 'TP3', 'TP4', 'P1H', 'P2H', 'P3H', 'P4H', 'PO1', 'PO2', 'PO5', 'PO6', 'O9', 'O10', 'FT7H', 'FT8H', 'TP7H', 'TP8H', 'PO9', 'PO10', 'IZ2', 'OZ2', 'PZ2', 'CPZ2', 'TPP9H', 'TPP10H', 'AFF1', 'AFF2', 'FFC5H', 'FFC3H', 'FFC4H', 'FFC6H', 'FCC5H', 'FCC3H', 'FCC4H', 'FCC6H', 'CCP5H', 'CCP3H', 'CCP4H', 'CCP6H', 'CPP5H', 'CPP3H', 'CPP4H', 'CPP6H', 'PPO1', 'PPO2', 'I1', 'I2', 'AFP3H', 'AFP4H', 'AFF5H', 'AFF6H', 'FFT7H', 'FFC1H', 'FFC2H', 'FFT8H', 'FTT9H', 'FTT7H', 'FCC1H', 'FCC2H', 'FTT8H', 'FTT10H', 'TTP7H', 'CCP1H', 'CCP2H', 'TTP8H', 'TPP7H', 'CPP1H', 'CPP2H', 'TPP8H', 'PPO9H', 'PPO5H', 'PPO6H', 'PPO10H', 'POO9H', 'POO3H', 'POO4H', 'POO10H', 'OI1H', 'OI2H', 'T3', 'T4', 'T9', 'T10', 'AFP1', 'AFP2', 'AFF1H', 'AFF2H', 'PPO1H', 'POO1', 'POO2', 'PPO2H', 'NONE', 'T5', 'T6']
electrode_names = [ch.upper() for ch in electrode_names]

electrode_num   = len(electrode_names)

electrode_none_idx  = electrode_names.index('NONE')

electrode_name2idx  = {v:i for i,v in enumerate(electrode_names)}

import torch

def raw_chs2chs_map(raw_chs):
    raw_chs = [i.upper() for i in raw_chs]
    chs_map = ([],[])
    for i,ch_name in enumerate(raw_chs):
        index = electrode_name2idx[ch_name]
        chs_map[0].append(index)
        chs_map[1].append(i)
    chs_map = (torch.tensor(chs_map[0], dtype=torch.long),
                torch.tensor(chs_map[1], dtype=torch.long))
    return chs_map

def raw_chs2chan_ids(raw_chs):
    # 将原始通道列表转换为集合（O(1)查找复杂度）
    raw_chs_set = set(raw_chs)  # 只执行一次O(n)转换
    
    # 使用列表推导式 + 集合成员检查（O(1)复杂度）
    chs = [
        electrode_name2idx[ch_name] if ch_name in raw_chs_set else electrode_none_idx
        for ch_name in electrode_names
    ]
    
    return torch.LongTensor(chs)
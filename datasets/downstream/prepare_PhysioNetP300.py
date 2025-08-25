import tqdm
import torch
import os
import mne

dataset_fold = "PhysioNetP300/"


all_chans = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

fmin=0
fmax=120
tmin=-0.1
tmax=2
for sub in [2,3,4,5,6,7,9,11]:

    #path = "erp-based-brain-computer-interface-recordings-1.0.0/files/s{:02d}".format(sub)
    path="erp-based-brain-computer-interface-recordings-1.0.0/files/erpbci/1.0.0/s{:02d}".format(sub) # based on the current state of the dataset
    for file in os.listdir(path):
        if not file.endswith(".edf"):continue
        raw = mne.io.read_raw_edf(os.path.join(path, file))
        raw.pick_channels(all_chans)
        
        events, event_id = mne.events_from_annotations(raw)
        
        event_map = {}
        tgt = None
        for k,v in event_id.items():
            if k[0:4]=='#Tgt':
                tgt = k[4]
            event_map[v] = k
        # assert event_map[1][0:4]=='#Tgt' and event_map[2]=='#end' and event_map[3]=='#start', event_map
        assert tgt is not None
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin = tmin, tmax=tmax,event_repeated='drop', preload=True, proj=False)#,event_repeated='drop',reject_by_annotation=True)
        epochs.filter(fmin, fmax,method = 'iir')
        epochs.resample(256)
        stims = [x[2] for x in epochs.events]
        # print(stims)
        data = epochs.get_data()
        for i,(d,t) in tqdm.tqdm(enumerate(zip(data, stims))):
            t = event_map[t]
            if t.startswith('#Tgt') or t.startswith('#end') or t.startswith('#start') or t[0]=='#':
                continue
            label = 1 if tgt in t else 0
            # -- save
            x = torch.tensor(d*1e3)
            y = label
            spath = os.path.join(dataset_fold, f'{y}')
            os.makedirs(spath, exist_ok=True)  # create the correct folder
            spath = os.path.join(spath, f'{i}.sub{sub}.pt')  # add .pt for PyTorch files
            torch.save(x, spath)

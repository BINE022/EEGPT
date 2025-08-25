TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml and save into the `datasets/downstream/tuh_eeg_abnormal` folder, which organized as:
```
datasets/downstream/tuh_eeg_abnormal/v3.0.1/edf/train/normal/01_tcp_ar/aaaaaqax_s001_t000.edf
...
datasets/downstream/tuh_eeg_abnormal/v3.0.1/edf/train/normal/01_tcp_ar/aaaaaaav_s004_t000.edf
...
datasets/downstream/tuh_eeg_abnormal/v3.0.1/edf/train/abnormal/01_tcp_ar/aaaaaaaq_s004_t000.edf
...
datasets/downstream/tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar/aaaaapys_s002_t001.edf
```
Then run the following command to preprocess the data:
```preprocess
cd downstream_tueg/dataset_maker
python make_TUAB.py
```

TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml and save into the `datasets/downstream/tuh_eeg_events` folder, which organized as:
```
datasets/downstream/tuh_eeg_events/v2.0.1/edf/train/aaaaafop/aaaaafop_00000001_ch021.lab
...
datasets/downstream/tuh_eeg_events/v2.0.1/edf/train/aaaaaaar/aaaaaaar_00000001.edf
...
datasets/downstream/tuh_eeg_events/v2.0.1/edf/eval/099/bckg_099_a_1.edf
```
Then run the following command to preprocess the data:
```preprocess
cd downstream_tueg/dataset_maker
python make_TUEV.py
```

BCIC-2A and BCIC-2B dataset is downloaded from https://www.bbci.de/competition/iv/#datasets and save into the `datasets/downstream/Raw_data` folder, which organized as:
```
datasets/downstream/Raw_data/BCICIV_2a_gdf/A01E.gdf
datasets/downstream/Raw_data/BCICIV_2a_gdf/A01E.mat
...
datasets/downstream/Raw_data/BCICIV_2a_gdf/A09T.gdf
datasets/downstream/Raw_data/BCICIV_2a_gdf/A09T.mat
...
datasets/downstream/Raw_data/BCICIV_2b_gdf/B0905E.gdf
datasets/downstream/Raw_data/BCICIV_2b_gdf/B0905E.mat
```
Then run the following command to preprocess the data:
```preprocess
cd downstream/Data_process
python process_function.py
```

KaggleERN datasets can be downloaded from https://www.kaggle.com/c/inria-bci-challenge/data and save into the `datasets/downstream/KaggleERN` folder, which organized as:
```
datasets/downstream/KaggleERN/TrainLabels.csv
datasets/downstream/KaggleERN/true_labels.csv
datasets/downstream/KaggleERN/train/Data_S02_Sess01.csv
...
datasets/downstream/KaggleERN/train/Data_S26_Sess05.csv
...
datasets/downstream/KaggleERN/test/Data_S01_Sess01.csv
...
datasets/downstream/KaggleERN/test/Data_S25_Sess05.csv
```

PhysioP300 datasets can be downloaded from https://physionet.org/content/erpbci/1.0.0/ and save into the `datasets/downstream/erp-based-brain-computer-interface-recordings-1.0.0` folder, which organized as:
```
datasets/downstream/erp-based-brain-computer-interface-recordings-1.0.0/files/erpbci/1.0.0/s01/rc01.edf
...
datasets/downstream/erp-based-brain-computer-interface-recordings-1.0.0/files/erpbci/1.0.0/s11/rc01.edf
...
Then run the following command to preprocess the data:
```preprocess
cd datasets/downstream
python prepare_PhysioNetP300.py
```

For preparing Sleep-EDF dataset, you can run the following command to preprocess the data:
```preprocess
cd datasets/downstream
python prepare_sleep.py
```

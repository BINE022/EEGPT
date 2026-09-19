## Pretraining Datasets Download

### PhysioNetMI

Follow this [link](https://www.physionet.org/content/eegmmidb/1.0.0/) to download the PhysioNetMI dataset.
Save the dataset in the `PhysionetMI` folder of the project and files should be organized as:

``` 
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S001/S001R01.edf
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S001/S001R01.edf.event
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S001/S001R02.edf
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S001/S001R02.edf.event
...
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S001/S001R14.edf
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S001/S001R14.edf.event
...
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S109/S109R14.edf
datasets/pretrain/PhysioNetMI/files/eegmmidb/1.0.0/S109/S109R14.edf.event
```
### TSUBenchmark

Download URL: http://bci.med.tsinghua.edu.cn/

In order to use this dataset, save all .mat files in `TSUBenchmark`, containing the following files:

``` 
Freq_Phase.mat
S1.mat
...
S35.mat
``` 
NOTE: Ensure there is no other type files (.pdf, .txt, etc) in the folder.

### M3CV

Download URL: https://aistudio.baidu.com/aistudio/datasetdetail/151025/0

In order to use this dataset, the download dataset folder aistudio is required, save it to 'aistudio' folder, containing the following files:

```
Calibration_Info.csv
Enrollment_Info.csv
Testing_Info.csv
Calibration (unzipped Calibration.zip)
Testing (unzipped Testing.zip)
Enrollment (unzipped Enrollment.zip)
```

### SEED

Download URL: https://bcmi.sjtu.edu.cn/home/seed/index.html

In order to use this dataset, the download folder Preprocessed_EEG is required, save it into `SEED` folder, containing the following files:
```
label.mat
readme.txt
1_20131027.mat
...
15_20131105.mat
```
NOTE: `SEED` must directly contain these files (the `Preprocessed_EEG` **content layer**). If the parent folder is used instead, `SEEDDataset` crashes with `invalid literal for int()` (it tries to parse `channel-order.xlsx` as a subject directory).

### HGD (High-Gamma Dataset)

Download URL: https://gin.g-node.org/robintibor/high-gamma-dataset

In order to use this dataset, save it into `high_gamma` folder, containing the following files:
```
high_gamma/train/1.edf ... high_gamma/train/14.edf
high_gamma/test/1.edf ... high_gamma/test/14.edf
```


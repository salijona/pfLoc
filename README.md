# Mobile Positioning and Trajectory Reconstruction Based on Mobile Phone Network Data: A Tentative Using Particle Filter

This work is accepted for publication in MT-ITS 2021.

<p align="center">
<img src="https://github.com/salijona/pfLoc/blob/main/pf_flow.png" class="center"></p>

In this paper, we demonstrated a method based on Particle Filter for user localization and trajectory reconstruction using CDR data. The method was evaluated both in synthentic and real case data provided by a mobile operator in Estonia. 

## Motivation
We are releasing our approach's source code for mobile positioning and trajectory reconstruction  with intention in contributing to advancement of state of the art in localization and trajectory reconstruction using mobile data. We hope it will serve  as well as an initiative in introducing new source of datasets to provide information about traffic status and travel information based on mobile phone network data such as CDR or VLR. 

## Usage 

### Environment 
Use the commands below to create a python environment named pf from environment.yml file in master branch using Anaconda. 
```
$ git checkout master
$ conda env create -f environment. yml
$ conda activate pf
```
### Dataset
We are providing the syntentic dataset that we generated using two samples (id = 162 and id = 197) from T-drive   dataset  that  contains  a  one-week  GPS  trajectory of  taxis  in  the  city  of  Beijing. The dataset was generated using a fixed cell size of 1800 m. The CDR events  are generated with the  assumption  that  when  a  GPS  event  is  triggered,  at  the same  time  a  CDR  event  is  triggered  automatically.  You will find in the master branch the Beijing_data folder with .csv files for CDR events and gps events for evaluation. 

### Experiments
Run the experiments by code access point in the file Particle_Filter_Hybrid.py located in src folder:
```
$ python3 Particle_Filter_Hybrid.py
```
or for Windows:
```
$ python Particle_Filter_Hybrid.py
```
### Visualization
The paths can be visualized using draw function in utils file or through a geographic information system application that supports viewing, editing, and analysis of geospatial data like QGIS. However due to draw function performing only basic markers we would advise on using tools like QGIS. To visualize the locations using QGIS you need to save the predicted locations in every iteration of the algorithm in a .csv file. 
Below is an example of localization and trajectory reconstruction using QGIS compared to actual GPS points. 

<p align="center">
<img src="https://github.com/salijona/pfLoc/blob/main/predicted_path_pf.PNG" width="600" height="300" class="center"></p>

## Licence 
This source code is released under a [GPLv3.0](https://github.com/simonwu53/NetCalib-Lidar-Camera-Auto-calibration/blob/master/LICENSE) license. 

For a closed-source version for commercial purposes, please contact the authors: [Dyrmishi](mailto:salijona.dyrmishi@uni.lu) and [Hadachi](mailto:hadachi@ut.ee)


## Contributors
Salijona Dyrmishi; Amnir Hadachi.  

## Citation 
If you use our source code or synthetic dataset in an academic work, please cite:
```
@inproceedings{Dyrmishi2021Mob,
  title={Mobile Positioning and Trajectory Reconstruction Based on Mobile Phone Network Data: A Tentative 
          Using Particle Filter},
  author={Salijona, Dyrmishi; Hadachi, Amnir},
  booktitle={Proceedings of the 7th International IEEE Conference on Models and Technologies for Intelligent
              Transportation Systems},
  year={2021},
  organization={IEEE}
}
```

Preprint version of the paper can be found [here]().



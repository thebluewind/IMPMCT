# IMPMCT-Integrated-Multi-source-Polar-Meso-Cyclone-Tracks
Here we will provide the main code for generating the IMPMCT dataset( https://doi.org/10.5281/zenodo.15113263 ).

Regarding the yolov8-obb-pose model, please refer to https://github.com/yzqxy/ultralytics-obb_segment.

The relevant model weights and validation sets are restored in https://doi.org/10.5281/zenodo.15119534.

1 identify.ipynb: This file is used to identify the vortex tracks in ERA5 data.
2 AVHRR_matched.ipynb: This file is used to match AVHRR data with the vortex tracks . 
3 VCI_create.ipynb: This file is used to create the VCI images for every vortex track if AVHRR data is available.
4 obb_pose_detect.ipynb: This file is used to detect the cyclone features in the VCI images using the yolov8-obb-pose model.
5 ASCAT_AVHRR_matched.ipynb: This file is used to match ASCAT\QUIKSCAT data with the cyclone features identified in the previous step.
6 example_data(folder) is the example dataset used in above codes.
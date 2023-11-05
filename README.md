# aero-vloc
aero-vloc is a tool for UAVs localization using different VPR systems and feature matchers.
VPR systems AnyLoc, CosPlace, EigenPlaces, MixVPR, NetVLAD are now supported as well as LightGlue and SuperGlue keypoint matchers.

## Weights
Weights for MixVPR, NetVLAD and SuperGlue as well as cluster centers for AnyLoc can be downloaded [here](https://drive.google.com/file/d/1JJWjbaY59XNICiXfQYdwoTYC6pIbzc_4/view?usp=sharing).
All other necessary files for CosPlace, EigenPlaces and SuperPoint will be downloaded automatically via TorchHub.

## Usage
Please check `example.ipynb` for an example of localizing and using the Recall metric.
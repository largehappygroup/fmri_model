import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from collections import Counter
import xml.etree.ElementTree as ET

# read in atlases
atlas_base_path = "/home/zachkaras/fmri_model/analysis/pipeline/atlases"

# read in 2d mni mask
mask = nib.load(f"{atlas_base_path}/MNI152_T1_2mm_brain_mask.nii.gz")
og_shape = mask.shape
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load(f"{atlas_base_path}/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx] # contains the schaefer parcel numbers
cortex_vx = np.where(atlas_only_brain != 0)[0]
parcel_nums = atlas_only_brain[cortex_vx]
hemis = ['Left' if ((parcel <= 200) & (parcel > 0)) else 'Right' for parcel in parcel_nums]

# Loading in atlas and labels for harvard-oxford atlas regions
hox_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
hox = nib.load(hox_data['filename']).get_fdata()
hox_vec = hox.flatten()
hox_only_brain = hox_vec[brain_idx]
hox_nums = hox_only_brain[cortex_vx] # Obtaining voxels in Harvard Oxford atlas that are in the schaefer atlas.
                                     #      The voxels in the Schaefer atlas that correspond to empty space in HarvOx atlas (values of 0) will get filtered out when we match to region names.
hox_nums = [int(i) for i in hox_nums]

hox_label_path = "/home/zachkaras/atlases/HarvardOxford-Cortical.xml"
hox_labels = [label.text for label in ET.parse(hox_label_path).getroot().iter('label')] # indices are the region numbers
hox_labels.insert(0,'None')

hox_region_names = [f"{hemis[i]} {hox_labels[hox_nums[i]]}" for i,region in enumerate(hox_nums)]
hox_region_counts = dict(Counter(hox_region_names))
del hox_region_counts['Left None']
del hox_region_counts['Right None']

with open("Harvard_Oxford_voxel_counts.pkl", 'wb') as f:
    pickle.dump(hox_region_counts, f)


"""import pickle

with open("schaefer_parcel_counts.pkl", 'rb') as f:
    voxel_counts = pickle.load(f)
    
with open("schaefer_parcel_labels.pkl", 'rb') as f:
    parcel_labels = pickle.load(f)
    
left_regions = [f"Left {region}" for region in set(parcel_labels['left'].values())]
right_regions = [f"Right {region}" for region in set(parcel_labels['right'].values())]
left_regions.extend(right_regions)
all_regions = left_regions

# iterate through HarvOx regions to find schaefer parcels in a given region
# increment region counts by each parcel's voxel counts
# save in a dictionary where each HarvOx region has a corresponding voxel count
HarvOx_voxel_counts = {region: 0 for region in all_regions}
for hemi, parcel_region in parcel_labels.items():
    for parcel,region in parcel_region.items():
        num_voxels = voxel_counts[parcel]
        HarvOx_voxel_counts[f"{hemi.capitalize()} {region}"] += num_voxels
"""

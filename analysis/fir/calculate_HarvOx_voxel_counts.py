import pickle

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

with open("Harvard_Oxford_voxel_counts.pkl", 'wb') as f:
    pickle.dump(HarvOx_voxel_counts, f)
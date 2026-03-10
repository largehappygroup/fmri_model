# load in libraries
import pickle
import nibabel as nib
from nilearn import datasets
import xml.etree.ElementTree as ET   

atlas_base_path = "/home/zachkaras/fmri_model/analysis/pipeline/atlases"

def load_atlases():    
    
    # Schaefer Atlas
    schaefer = nib.load(f"{atlas_base_path}/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
    schaefer = schaefer.get_fdata()
    
    # Load in Harvard-Oxford atlas
    ho_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho = nib.load(ho_data['filename']).get_fdata()

    # Loading in labels for harvard-oxford atlas regions
    ho_labels = "/home/zachkaras/atlases/HarvardOxford-Cortical.xml"
    labels = [label.text for label in ET.parse(ho_labels).getroot().iter('label')] # indices are the region numbers - need to add one though

    return schaefer, ho, labels


def find_overlapping_regions(x_range,y_range,z_range, schaefer, ho):
    
    parcel_maps = dict()

    # iterating through x,y,z coordinates of schaefer atlas
    # then finding corresponding Harvard Oxford region numbers
    # sorting by the HO regions with the higest number of overlapping voxels
    for x in range(x_range):
        for y in range(y_range):
            for z in range(z_range):
                val = int(schaefer[x,y,z])
                
                if val != 0:
                    ho_equivalent = int(ho[x,y,z])
                    
                    if val not in parcel_maps.keys():
                        parcel_maps[val] = dict()
                    
                    if ho_equivalent not in parcel_maps[val].keys():
                        parcel_maps[val][ho_equivalent] = 1
                    else:
                        parcel_maps[val][ho_equivalent] += 1

    # Don't think it needs to be sorted
    # sorted_parcel_maps = dict(sorted(parcel_maps.items(), key=lambda x: x[0]))
    return parcel_maps


def map_to_then_group_by_region_names(parcel_maps, labels):
    schaefer_ho_map = dict()
    for sch,hos in parcel_maps.items():
        sorted_hos = dict(sorted(hos.items(), key=lambda x: x[1], reverse=True))
        top_ho = list(sorted_hos.keys())[0]
        ho_region = labels[top_ho-1]
        schaefer_ho_map[sch] = ho_region

    sorted_ho_map = dict(sorted(schaefer_ho_map.items(), key=lambda x: labels.index(x[1]))) 
    return sorted_ho_map

def split_hemisphere_regions(region_groupings):

    by_hemisphere = {'left' : {}, 'right' : {}}
    for sch,reg in region_groupings.items():
        
        hemi = 'right' if sch > 200 else 'left'

        by_hemisphere[hemi][sch] = reg

    return by_hemisphere

                
def main():

    schaefer,ho,labels = load_atlases()

    x_range,y_range,z_range = schaefer.shape
    
    parcel_maps = find_overlapping_regions(x_range, y_range, z_range, schaefer, ho)

    region_groupings = map_to_then_group_by_region_names(parcel_maps, labels)

    by_hemisphere = split_hemisphere_regions(region_groupings)


    with open("schaefer_parcel_labels.pkl", 'wb') as f:
        pickle.dump(by_hemisphere, f)

if __name__=="__main__":
    main()
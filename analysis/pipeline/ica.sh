#/bin/bash
# after preprocessing 

perform_ica(){
      melodic -i "$1/raw_mc_epi2mni.nii.gz" -d 60 -o "$1/filtered_func_data.ica/" --Oorig --report --tr=0.8 -v 
}

format_for_fix(){
      echo "formatting output directory for FIX"
      mkdir "$1/mc"
      mkdir "$1/reg"
      
      mv "$1/st_mc.nii.par" "$1/mc/prefiltered_func_data_mcf.par"
      fslroi "$1/raw_mc_epi2mni.nii.gz" "$1/reg/example_func.nii.gz" 0 1

      # create mask for 4D functional data
      flirt -in "$1/BrainExtractionMask.nii.gz" -ref "$1/reg/example_func.nii.gz" -out "$1/mask.nii.gz" -applyxfm -usesqform
      
      # create temporal mean of 4d data
      fslmaths "$1/raw_mc_epi2mni.nii.gz" -Tmean "$1/mean_func.nii.gz"

      # renaming preprocessed fMRI file
      mv "$1/raw_mc_epi2mni.nii.gz" "$1/filtered_func_data.nii.gz"
      
      # move example anatomical to reg/highres.nii.gz
      mv "$1/BrainExtractionBrain.nii.gz" "$1/reg/highres.nii.gz"
      flirt -in "$1/reg/highres.nii.gz" -ref "$1/reg/example_func.nii.gz" -out "$1/reg/highres2example_func" -omat "$1/reg/highres2example_func.mat"

}

perform_fix(){
      echo "Performing fix to classify components"
      ~/fix/fix -c "$1" Loop_ML_Model.RData 30
}

remove_components(){
      echo "Removing components"
      { # try
            ~/fix/fix -a "$1/fix4melview_Loop_ML_Model_thr30.txt"
      } || { # catch
            echo "There may be an issue with the data file"
            fslhd "$1/raw_mc_epi2mni.nii.gz" >> "$1/$2_header.txt"
      }
}

# for loop for going through output directories  
find "/home/zachkaras/fmri/fmri_model_data/midprocess_prose" -maxdepth 1 -type d | while read -r folder; do
      foldername="${folder:54}"
      #echo "$foldername"
      if [[ "$foldername" =~ ^[0-9]{3}$ ]]; then
            echo "$foldername"
            perform_ica "$folder"
            format_for_fix "$folder"
            perform_fix "$folder"
            remove_components "$folder" "$foldername"
            
            #break
            #((counter++))
            #if [[ $counter -ge 3 ]]; then
            #      break
            #fi
            #break
      fi
done



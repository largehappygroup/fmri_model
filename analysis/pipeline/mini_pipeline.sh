#/bin/bash

preprocess(){
      # need to create a reference volume
      fslroi $1 "$2"/reference.nii.gz 0 1
     
      # need to register 
      # register the reference volume to the template
      antsRegistrationSyNQuick.sh -d 3 \
            -f $3 \
            -m "$2"/reference.nii.gz \
            -o "$2"/affine_ \
            -t r

      # apply that registration to the full functional scan
      
}


# main function to iterate through participants' fmri files

datadir='/home/zachkaras/fmri/fmri_model_data/bids_formatted_clean/derivatives/fmriprep-v24'

find $datadir -maxdepth 1 -type d | while read folder; do
      idnum="${folder:83}"
      if echo $idnum | grep -E -q '^sub-[0-9]{3}'; then
            echo $idnum
            nifti="${datadir}/${idnum}/ses-001/func/${idnum}_task-coding_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"

            # rename file 
            mv $nifti "${datadir}/${idnum}/ses-001/func/${idnum}_fsl_mni_2mm.nii.gz"
            input="${datadir}/${idnum}/ses-001/func/${idnum}_fsl_mni_2mm.nii.gz"

            atlas="/home/zachkaras/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz"
            outputdir="/home/zachkaras/fmri/fmri_model_data/bids_formatted_clean/derivatives/fmriprep-v24/${idnum}/ses-001/func"
            echo "working on registering $idnum"
            #echo "preprocessing $input to $outputdir with $atlas"
            preprocess $input $outputdir $atlas
            break
      fi
done




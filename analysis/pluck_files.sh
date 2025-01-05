#!/bin/bash


# find files in codesynth folder
DATADIR="/storage2/fmridata/fmri-data-codesynth/"
find "$DATADIR" -maxdepth 1 -type d | while read -r folder; do
	FOLDERNAME="${folder:39}"
	if echo "$FOLDERNAME" | grep -E -q '^[0-9]{3}$'; then
		echo "$FOLDERNAME"
		NEWDIR="/home/zachkaras/fmri/fmri_model_data/raw/$FOLDERNAME/"
		mkdir "$NEWDIR"
		ANAT="$folder/fmri-scan/anatomy/t1spgr_208sl/ht1spgr_208sl.nii"
		FUNC="$folder/fmri-scan/func/frcode/run_01/utrun_01.nii"

		cp "$ANAT" "$NEWDIR"
		cp "$FUNC" "$NEWDIR"
		#break
	fi
done



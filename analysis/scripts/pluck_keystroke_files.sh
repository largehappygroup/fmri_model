#!/bin/bash


# find files in codesynth folder
DATADIR="/storage2/fmridata/fmri-data-codesynth/"
find "$DATADIR" -maxdepth 1 -type d | while read -r folder; do
	FOLDERNAME="${folder:39}"
	if echo "$FOLDERNAME" | grep -E -q '^[0-9]{3}$'; then
		echo "$FOLDERNAME"
		NEWDIR="/home/zachkaras/fmri/fmri_model/data/$FOLDERNAME"
		mkdir "$NEWDIR"

        ls $folder/*.txt
        cp $folder/*.txt $NEWDIR
	fi
done

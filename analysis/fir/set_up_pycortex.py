import cortex

ref = "/home/zachkaras/atlases/MNI152_T1_2mm_brain.nii.gz"

cortex.align.automatic_fsl(subject='fsaverage', xfmname='mni2py', reference=ref)
# cortex.align.automatic(subject='fsaverage', xfmname='mni2py', reference=ref)
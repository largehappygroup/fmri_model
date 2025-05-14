function csfMRIv2_segmentStrip(hiresSeq)
% Segmentation and skull-stripping of hi-res anatomical image.
%
% FORMAT csfMRIv2_segmentStrip(hiresSeq)
%
% REQUIRED INPUT:
%   hiresSeq
%       String specifying which hires image to segment, either 'spgr'
%       (spoiled gradient recall echo) or 'flair' (fluid-attenuated
%       inversion recovery). I think the SPGR image works a little better
%       for subsequent coregisttering, but I'm not convinced there's a 
%       massive difference in overall segmentation quality.
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Tell SPM where the hires scan lives.
%--------------------------------------------------------------------------
%hiresPath = ['/wrw*/anatomy/t1' hiresSeq '*'];
    wr = dir('wrw*');
    wrw = wr(1).name;
    hiresPath = [''];
    switch hiresSeq
        case 'spgr'
            hiresPath = ['./' wrw '/anatomy/t1spgr_208sl']
            matlabbatch{1}.spm.spatial.preproc.channel.vols = cellstr(strcat([hiresPath '/'], spm_select('List', hiresPath, ['^t1spgr_208sl.*nii$'])));
        case 'flair'
            hiresPath = ['./' wrw '/anatomy/t1spgr_208sl']
            matlabbatch{1}.spm.spatial.preproc.channel.vols = cellstr(strcat([hiresPath '/'], spm_select('List', hiresPath, ['^t1overlay_60sl.*nii$'])));
    end
    
    
        
% Set parameters for bias field correction.
%--------------------------------------------------------------------------
% This deals with the fact that magnetic field inhomogeneities can cause
% slight variations in image contrast.

   matlabbatch{1}.spm.spatial.preproc.channel.biasreg  = 0.001;
   matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
   matlabbatch{1}.spm.spatial.preproc.channel.write    = [0 1];
        
% Define tissue probability maps.
%--------------------------------------------------------------------------
% These are a priori estimates of how likely it is that a voxel is grey
% matter, white matter, CSF, etc.

    matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm    = cellstr(fullfile(spm('Dir'), 'tpm', 'TPM.nii,1'));
    matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus  = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [1 1];
    
    matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm    = cellstr(fullfile(spm('Dir'), 'tpm', 'TPM.nii,2'));
    matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus  = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [1 1];
    
    matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm    = cellstr(fullfile(spm('Dir'), 'tpm', 'TPM.nii,3'));
    matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus  = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [1 1];
    
    matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm    = cellstr(fullfile(spm('Dir'), 'tpm', 'TPM.nii,4'));
    matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus  = 3;
    matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
    
    matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm    = cellstr(fullfile(spm('Dir'), 'tpm', 'TPM.nii,5'));
    matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus  = 4;
    matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
    
    matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm    = cellstr(fullfile(spm('Dir'), 'tpm', 'TPM.nii,6'));
    matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus  = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
        
% Set parameters for the estimation of deformation maps.
%--------------------------------------------------------------------------
% Here we'll get both forward deformations (mapping subject space to MNI) 
% and inverse deformations (MNI to subject). The voxels in these images 
% contain 3D tensors that define the spatial warp from one image to
% another.
        
    matlabbatch{1}.spm.spatial.preproc.warp.mrf     = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg     = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg  = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm    = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp    = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write   = [1 1];
        
% Run segmentation.
%--------------------------------------------------------------------------
    
    spm_jobman('run', matlabbatch);
        
% Grab grey, white, and CSF segments.
%--------------------------------------------------------------------------
    
    gm  = cellstr(strcat([hiresPath '/'], spm_select('List', hiresPath, '^c1t*.*nii$')));
    wm  = cellstr(strcat([hiresPath '/'], spm_select('List', hiresPath, '^c2t*.*nii$')));
    csf = cellstr(strcat([hiresPath '/'], spm_select('List', hiresPath, '^c3t*.*nii$')));
        
% Get corrected hires.
%--------------------------------------------------------------------------
% Here, the 'm' prefix indicates a MODULATED (i.e. bias regularised) image, 
% per SPM's convention.
    
    biasCorr = cellstr(strcat([hiresPath '/'], spm_select('List', hiresPath, '^mt1*.*nii$')));
        
% Skull-strip the T1 using ImCalc.
%--------------------------------------------------------------------------
% The grey, white, and CSF segments are essentially binary 'masks' that
% define the various components of brainspace, so we add them together and
% apply it to the hires in order to get rid of all non-brain voxels.
    
    [~,name,~] = fileparts(char(biasCorr));
        
    setDefaultsSPM;
    
    matlabbatch = {};

    matlabbatch{1}.spm.util.imcalc.input          = cellstr([gm; wm; csf; biasCorr]);

    matlabbatch{1}.spm.util.imcalc.output         = [hiresPath '/ss-' name '.nii'];
    matlabbatch{1}.spm.util.imcalc.expression     = '(i1 + i2 + i3) .* i4';

    matlabbatch{1}.spm.util.imcalc.options.dmtx   = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask   = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype  = 4;

    spm_jobman('run',matlabbatch);
    
end

%-------------------------------------------------------------------------%
% BEGIN SUBROUTINES                                                       %
%-------------------------------------------------------------------------%

% Initialize default parameters for SPM.
%-------------------------------------------------------------------------%
function setDefaultsSPM

    spm('defaults','fMRI');
    warning off MATLAB:FINITE:obsoleteFunction;
    spm_jobman('initcfg');
    
end
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% END SUBROUTINES                                                         %
%-------------------------------------------------------------------------%
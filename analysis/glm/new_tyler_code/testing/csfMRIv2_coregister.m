function csfMRIv2_coregister(nRun, hiresSeq)
% Coregistration of functional and anatomical images.
%
% FORMAT csfMRIvs_coregister(nRun, hiresSeq)
%
%   REQUIRED INPUT:
%       nRun
%           Number of scanning runs for this subject.
%
%       hiresSeq
%           Which hires to coregister to, either 'spgr' or 'flair'.
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Specify REFERENCE SCAN (i.e. the thing that stays still).
%--------------------------------------------------------------------------
% Here, we want this to be the bias-regularised, skull-stripped hires 
% image. We choose  this because we previously estimated a FORWARD 
% DEFORMATION FIELD during segmentation that defines the warp to MNI space.
% Thus, we need to map the functional data to the hires so those parameters
% are valid for subsequent normalization. Note I've assumed the file begins
% with 'ss-mt1' (skull-stripped, bias-corrected T1).

    switch hiresSeq
        case 'spgr'
            matlabbatch{1}.spm.spatial.coreg.estimate.ref = cellstr(strcat([pwd '/anatomy/t1spgr_208sl/'], spm_select('List', [pwd '/anatomy/t1spgr_208sl/'], '^ss-mt1*.*nii$')));
        case 'flair'
            matlabbatch{1}.spm.spatial.coreg.estimate.ref = cellstr(strcat([pwd '/anatomy/t1overlay_60sl/'], spm_select('List', [pwd '/anatomy/t1overlay_60sl/'], '^ss-mt1*.*nii$')));
    end
    
% Specify SOURCE SCAN (i.e. the thing we're mapping onto the reference
% image).
%--------------------------------------------------------------------------
% This is the mean functional scan after we realign/unwarp. We assume the 
% file begins with 'meanu' (mean unwarped image). Note that we also need to
% specify the other functional data so the estimated coregistration 
% parameters can be added to those image headers.

    matlabbatch{1}.spm.spatial.coreg.estimate.source = cellstr(strcat([pwd '/func/mean/'], spm_select('List', [pwd '/func/mean/'], '^meanu.*nii$')));
    
    allEPI = cellstr(strcat([pwd '/func/rest/run_01/'], spm_select('List', [pwd '/func/rest/run_01/'], 'urun_01.nii')));
    
    for iRun = 1:nRun
        
        scans  = cellstr(strcat([pwd '/func/reviewblock/run_0' num2str(iRun) '/'], spm_select('List', [pwd '/func/reviewblock/run_0' num2str(iRun) '/'], ['urun_0' num2str(iRun) '.nii'])));
        
        allEPI = [allEPI; scans];
        
    end
    
    matlabbatch{1}.spm.spatial.coreg.estimate.other             = allEPI;
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep      = [4 2];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm     = [7 7];
    
% Run coregistration.
%--------------------------------------------------------------------------

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
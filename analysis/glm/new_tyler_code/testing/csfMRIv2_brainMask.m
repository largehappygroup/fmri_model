function csfMRIv2_brainMask(hiresSeq, smoothMask)
% Brain mask construction.
%
% FORMAT csfMRIv2_brainMask(hiresSeq, smoothMask)
%
% REQUIRED INPUT:
%   hiresSeq
%       String specifying which hires segments to use, either 'spgr' or
%       'flair'.
%
%   smoothMask
%       Apply a spatial smooth to the brainmask? 1 for YES, 0 for NO. This
%       doesn't necessarily make a huge difference but I smooth by default.
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Tell SPM where the hires data live, get the (normalized) grey and white
% matter segments.
%--------------------------------------------------------------------------
    hiresPath = [''];
    switch hiresSeq
        case 'spgr'
            hiresPath = [pwd '/anatomy/t1spgr_208sl']
        case 'flair'
            hiresPath = [pwd '/anatomy/t1overlay_60sl']
    end
    
    graySeg   = [hiresPath '/' spm_select('List', hiresPath, '^mwc1t1*.*nii$')];
    whiteSeg  = [hiresPath '/' spm_select('List', hiresPath, '^mwc2t1*.*nii$')];
    
% Create a new directory in which to store the brainmask.
%--------------------------------------------------------------------------

    mkdir([hiresPath '/Mask']);
    cd([ hiresPath '/Mask']);
    
% Specify inputs to ImCalc and compute mask.
%--------------------------------------------------------------------------
% The idea here is that we're defining our 'search space' for subsequent
% analyses - because we run a GLM on every voxel in the brain, we don't
% want to waste time specifying models and looking for 'activity' in voxels
% that are just CSF or somewhere outside the brain. Technically we should
% only see activity in grey matter (because that's where the neurons
% themselves are), but spatial smoothing of the fMRI data can cause signal
% to 'blur' into white matter voxels, so we include those here as well.

    matlabbatch{1}.spm.util.imcalc.input      = cellstr([graySeg; whiteSeg]);
    matlabbatch{1}.spm.util.imcalc.output     = 'brainmask.nii';
    matlabbatch{1}.spm.util.imcalc.expression = '(i1 > 0.05) | (i2 > 0.05)';

    matlabbatch{1}.spm.util.imcalc.options.dmtx   = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask   = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype  = 4;
    
    spm_jobman('run',matlabbatch);
    
% Smooth if desired.
%--------------------------------------------------------------------------

    if smoothMask
        
        setDefaultsSPM;
        
        matlabbatch = {};
        
        matlabbatch{1}.spm.spatial.smooth.data   = cellstr([pwd '/brainmask.nii']);
        matlabbatch{1}.spm.spatial.smooth.fwhm   = [5 5 5];
        matlabbatch{1}.spm.spatial.smooth.dtype  = 0;
        matlabbatch{1}.spm.spatial.smooth.im     = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = 's';
    
        spm_jobman('run',matlabbatch);
        
    end
    
    cd ..
    cd ..
    cd ..
    
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
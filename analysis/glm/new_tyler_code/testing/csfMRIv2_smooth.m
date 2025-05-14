function csfMRIv2_smooth(nRun)
% Spatial smoothing
%
% FORMAT csfMRIv2_smooth(nRun)
%
%   REQUIRED INPUT:
%       nRun
%           Number of scanning runs for this subject.
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Get all the normalized, realigned/unwarped functional 
%--------------------------------------------------------------------------

    allEPI = cellstr(strcat([pwd '/func/rest/run_01/'], spm_select('List', [pwd '/func/rest/run_01/'], '^wurun*.*nii$')));
    
    for iRun = 1:nRun
        
        scans  = cellstr(strcat([pwd '/func/reviewblock/run_0' num2str(iRun) '/'], spm_select('List', [pwd '/func/reviewblock/run_0' num2str(iRun) '/'], '^wurun*.*nii$')));
        
        allEPI = [allEPI; scans];
        
    end
    
    meanScan = cellstr(strcat([pwd 'func/mean/'], spm_select('List', [pwd 'func/mean'], '^wmeanu*.*nii$')));
    allEPI   = [allEPI; meanScan];
        
% Specify smoothing parameters and run.
%--------------------------------------------------------------------------
% We're using a 5mm^3 FWHM Gaussian smoothing kernel. This is also
% something we could tweak if we wanted. Recall that smoothing is
% essentially just a cheap trick to boost our signal-to-noise ratio - if
% noise is indeed Gaussian/iid across the brain, this will push the signal
% in noisy voxels down to zero while retaining the actual task-related
% signals we're interested in. However, this works best if we know
% something about the nature of the signal we're trying to recover. Too
% little smoothing will result in too much noise leftover; too much
% smoothing can either average in a bunch of irrelevant noise, or cause
% spatially-distinct signals to 'fuse'. We typcally choose a kernel size
% that's no more than 2-3x the width of our voxels.

    matlabbatch{1}.spm.spatial.smooth.data   = allEPI;
    matlabbatch{1}.spm.spatial.smooth.fwhm   = [5 5 5];
    matlabbatch{1}.spm.spatial.smooth.dtype  = 0;
    matlabbatch{1}.spm.spatial.smooth.im     = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 's';
    
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
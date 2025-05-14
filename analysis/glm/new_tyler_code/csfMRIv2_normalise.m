function csfMRIv2_normalise(nRun, hiresSeq)
% Spatial normalization to MNI space.
%
% FORMAT csfMRIv2_normalise(nRun, hiresSeq)
%
%   REQUIRED INPUT:
%       nRun
%           Number of scanning runs for this subject.
%
%       hiresSeq
%           Which hires to normalize, either 'spgr' or 'flair'.
%__________________________________________________________________________


% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Normalise the (skull-stripped, bias-corrected) hires first.
%--------------------------------------------------------------------------

wr = dir('wrw*');
wrw = wr(1).name;
switch hiresSeq
        case 'spgr'
            matlabbatch{1}.spm.spatial.normalise.write.subj.def = cellstr(strcat([pwd '/' wrw '/anatomy/t1spgr_208sl/'], spm_select('List', [pwd '/' wrw '/anatomy/t1spgr_208sl/'], '^y_t1*.*nii$')));
            matlabbatch{1}.spm.spatial.normalise.write.subj.resample = cellstr(strcat([pwd '/' wrw '/anatomy/t1spgr_208sl/'], spm_select('List', [pwd '/' wrw '/anatomy/t1spgr_208sl/'], '^ss-mt1*.*nii$')));
        case 'flair'
            matlabbatch{1}.spm.spatial.normalise.write.subj.def = cellstr(strcat([pwd '/' wrw '/anatomy/t1overlay_60sl/'], spm_select('List', [pwd '/' wrw '/anatomy/t1overlay_60sl/'], '^y_t1*.*nii$')));
            matlabbatch{1}.spm.spatial.normalise.write.subj.resample = cellstr(strcat([pwd '/' wrw '/anatomy/t1overlay_60sl/'], spm_select('List', [pwd '/' wrw '/anatomy/t1overlay_60sl/'], '^ss-mt1*.*nii$')));
    end
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb     = [-78 -112 -70; 78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox    = [1 1 1];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
    
    spm_jobman('run',matlabbatch);
    
% Reset for the functional 
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Collect all the (realigned/unwarped) functional scans.
%--------------------------------------------------------------------------

    allEPI = cellstr(strcat([pwd '/' wrw '/func/rest/run_01/'], spm_select('List', [pwd '/' wrw '/func/rest/run_01/'], 'u
    run*.*nii$')));
    
    for iRun = 1:nRun
        
        scans  = cellstr(strcat([pwd '/' wrw '/func/typing/run_0' num2str(iRun) '/'], spm_select('List', [pwd '/' wrw '/func/typing/run_0' num2str(iRun) '/'], '^urun*.*nii$')));
        
        allEPI = [allEPI; scans];
        
    end
    
    meanScan = cellstr(strcat([pwd '/' wrw '/func/mean/'], spm_select('List', [pwd '/' wrw '/func/mean/'], '^meanu*.*nii$')));
    allEPI   = [allEPI; meanScan];
    
% Define normalization parameters and run.
%--------------------------------------------------------------------------
% Note that there is a vector here to specify voxel sizes (there's one
% above, too, but 1mm^3 voxels are standard for hi-res anatomicals) - if we
% wanted, we could resample the data to a slightly higher or lower
% resolution (e.g. 2mm^3 or 3mm^3), but I've kept the 'native' resolution
% of these scans. The reason is that anytime we're forced to reslice the
% data in some way, we have to interpolate, so we're effectively 
% abstracting farther and farther away from the 'true' signal in a given
% voxel. This has the potential to induce artifacts, so I'd rather just
% stick to the resolution we've got (this used to be somewhat unavoidable
% because slices were thicker, so voxels weren't entirely cubic).
    switch hiresSeq
        case 'spgr'
            matlabbatch{1}.spm.spatial.normalise.write.subj.def = cellstr(strcat([pwd '/' wrw '/anatomy/t1spgr_208sl/'], spm_select('List', [pwd '/' wrw '/anatomy/t1spgr_208sl/'], '^y_t1*.*nii$')));
        case 'flair'
            matlabbatch{1}.spm.spatial.normalise.write.subj.def = cellstr(strcat([pwd '/' wrw '/anatomy/t1overlay_60sl/'], spm_select('List', [pwd '/' wrw '/anatomy/t1overlay_60sl/'], '^y_t1*.*nii$')));
    end
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample   = allEPI;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb     = [-78 -112 -70; 78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox    = [2.4 2.4 2.4];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
    
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
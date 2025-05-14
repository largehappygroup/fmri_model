function csfMRIv3_smoothCons(subID, estMethod)

% Initialize default SPM configurations for fMRI.

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Get all the cons for this task.

    cd(['results.glm.' estMethod]);
    allCons = cellstr(strcat([pwd '/'], ...
        spm_select('List', [pwd '/'], ['^con*.*nii$'])));
        
% Specify smoothing parameters and run.

    matlabbatch{1}.spm.spatial.smooth.data   = allCons;
    matlabbatch{1}.spm.spatial.smooth.fwhm   = [5 5 5];
    matlabbatch{1}.spm.spatial.smooth.dtype  = 0;
    matlabbatch{1}.spm.spatial.smooth.im     = 1;
    matlabbatch{1}.spm.spatial.smooth.prefix = 's';
    
    spm_jobman('run',matlabbatch);
    
% Copy smoothed cons to relevant directory for second-level models.
    mkdir('results.smooth/');
    targetDir = [pwd '/results.smooth/'];
    
    % this needs to be modifed to a for loop. I don't know why Tyler had it just set for one.
    unix(['cp ' pwd '/scon_0001.nii ' ...
          targetDir subID '-con_0001.nii']);
    
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
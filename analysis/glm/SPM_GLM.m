% GLM from Tyler's SPM code
function SPM_GLM(subID, condition, fmri_datapath, outpath, onset_file, task)

    setDefaultsSPM;
    
    matlabbatch = {};

    duration = 60; % Duration for Long Response code and prose was 60s

%% TODO
% Unzip fMRI files and mask files, then zip them again after GLM

%% Create a new directory where we can store the relevant model output.
%--------------------------------------------------------------------------
    mkdir(outpath);
    cd(outpath);
    
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.dir = cellstr(pwd);
    
% Specify basic timing parameters.
%--------------------------------------------------------------------------
% Here we want to make sure we're specifying the appropriate units for time
% (seconds) and the TR of our pulse sequence (800ms). The other parameters
% are SPM's defaults - they basically just determine how to 'magnify' the
% temporal resolution of the 

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.timing.units   = 'secs';
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.timing.RT      = 0.800;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.timing.fmri_t  = 16;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.timing.fmri_t0 = 8;

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).scans = cellstr(fmri_datapath);
    onsets = readtable(onset_file);

    if size(onsets,2) > 1
        onsets = onsets.Var2;
    else
        onsets = onsets.Var1;
    end

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).name     = char(condition);
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).onset    = onsets;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).duration = duration;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).tmod     = 0;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).orth     = 1;

    % No parametric modulation
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});

% Specify hrf basis, mask, and noise model for objective function.
%--------------------------------------------------------------------------
% We use the canonical hrf (as opposed to a finite impulse response, for
% example) with no temporal or dispersion derivatives. One potential
% advantage to adding derivates is to account for small variations in the
% latency/width of the hrf. However, this can complicate contrast
% estimation, so we kept it simple here.
    mask_location = "/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/MNI152_T1_2mm_brain_mask.nii";
    % if task == "code"
    %     mask_location = sprintf("/home/zachkaras/fmri/fmri_model_data/midprocess/%s/mask.nii.gz", subID);
    % elseif task == "prose"
    %     mask_location = sprintf("/home/zachkaras/fmri/fmri_model_data/midprocess_prose/%s/mask.nii", subID);
    % end

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.fact             = struct('name', {}, 'levels', {});
    % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.bases.hrf.derivs = [1 1]; % ZK adding time and dispersion derivatives
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.volt             = 1;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.global           = 'None';
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mthresh          = 0.2;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mask             = cellstr(mask_location);
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.cvi              = 'wls';

    
    
% Run design spec.
%--------------------------------------------------------------------------
    % disp(matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).scans); % Adjust path if using standard GLM
    % disp(matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).name);
    spm_jobman('run', matlabbatch);
    
% Run model estimation.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_est.spmmat           = cellstr([pwd '/SPM.mat']);
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_est.method.Classical = 1;
    
    spm_jobman('run', matlabbatch);
    
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












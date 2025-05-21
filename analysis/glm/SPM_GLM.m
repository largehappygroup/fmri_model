% GLM from Tyler's SPM code
function SPM_GLM(subID, condition, fmri_datapath, outpath, onset_file, task)

    setDefaultsSPM;
    
    matlabbatch = {};

    duration = 60; % Duration for Long Response code and prose was 60s


% Create a new directory where we can store the relevant model output.
%--------------------------------------------------------------------------
    mkdir(outpath);
    cd(outpath);

    % location = pwd;
    % wr = dir('wrw*');
    % wrw = wr(1).name; 
    % cd ..;
    % parloc = pwd;
    % cd(location);
    % 
    % if paraMod
    %     mkdir('results.glm.rwls/');
    %     cd('results.glm.rwls/');
    %     load([parloc '/tsDifficulty.mat']);
    % else 
    %     unix('rm -r *rwls');
    %     mkdir('results.glm.rwls/');
    %     cd('results.glm.rwls/');
    % end
    
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

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).name     = condition;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).onset    = onsets;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).duration = duration;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).tmod     = 0;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).orth     = 1;

    % No parametric modulation
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(1).cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});

    % for iRun = 1:nRun
    % 
    %     % Point SPM to where the data live.
    % 
    %     % for each condition, read in corresponding nifti file 
    %     % mine will be from clean and clean_prose
    % 
    %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).scans = cellstr(strcat([location '/' wrw '/func/typing/run_0' num2str(iRun) '/'], spm_select('List', [location '/' wrw '/func/typing/run_0' num2str(iRun) '/'], '^wurun*.*nii$')));
    % 
    %     % nifti file for code in
    %     % /home/zachkaras/fmri/fmri_model_data/clean/{id_num}.nii.gz
    %     % nifti file for prose in
    %     % /home/zachkaras/fmri/fmri_model_data/clean_prose/{id_num}.nii.gz
    % 
    %     % durations will be 60s for everything
    % 
    %     % blocks for code writing experiment are only one condition
    % 
    %     % need to make onset files for loops/nonloops
    %     % then each question individually
    % 
    %     % duration will be trial durations (60s)
    % 
    %     % 
    % 
    % 
    % 
    %     % Mental.
    %     % generate_csv([location '/codesynth-block' num2str(iRun) '-final-151-1.txt'], iRun, 'M', 'onset');
    % 
    %     mentalOnset = csvread([location '/mentalOnset-' num2str(iRun) '.csv'], 1, 0);
    % 
    %     switch durModel
    %         case 'rt'
    %            % generate_csv([location '/codesynth-block' num2str(iRun) '-final-151-1.txt'], iRun, 'M', 'RT');
    %             mentalRT                = csvread([location '/mentalRT-' num2str(iRun) '.csv'], 1, 0);
    %             mentalRT(mentalRT == 0) = 30;
    %         case 'full'
    %             mentalRT = 30;
    %     end
    % 
    %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).name     = 'Mental';
    %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).onset    = mentalOnset;
    %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).duration = mentalRT;
    %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).tmod     = 0;
    %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).orth     = 1;
    % 
    %      % if paraMod
    %      % 
    %      %    % generate_csv([location '/codesynth-block' num2str(iRun) '-final-151-1.txt'], iRun, 'M', 'stimuli');
    %      %    mentalStim = csvread([location '/mentalStim-' num2str(iRun) '.csv'], 1, 0); 
    %      %    vals       = tsDifficulty(mentalStim, 2);
    %      % 
    %      %    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod.name  = 'mentalDifficulty';
    %      %    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod.param = vals - mean(vals);
    %      %    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod.poly  = 1;
    % 
    %     % else
    % 
    %         matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod       = struct('name', {}, 'param', {}, 'poly', {});
    % 
    %      % end
    % 
    %     %  % Tree.
    %     % 
    %     % % generate_csv([location '/codesynth-block' num2str(iRun) '-final-151-1.txt'], iRun, 'T', 'onset');
    %     % treeOnset = csvread([location '/treeOnset-' num2str(iRun) '.csv'], 1, 0);
    %     % 
    %     % switch durModel
    %     %     case 'rt'
    %     %         % generate_csv([location '/codesynth-block' num2str(iRun) '-final-151-1.txt'], iRun, 'T', 'RT');
    %     %         treeRT              = csvread([location '/treeRT-' num2str(iRun) '.csv'], 1, 0);
    %     %         treeRT(treeRT == 0) = 30;
    %     %     case 'full'
    %     %         treeRT = 30;
    %     % end
    %     % 
    %     % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).name     = 'Tree';
    %     % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).onset    = treeOnset;
    %     % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).duration = treeRT;
    %     % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).tmod     = 0;
    %     % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).orth     = 1;
    %     % 
    %     % if paraMod
    %     % 
    %     %     % generate_csv([location '/codesynth-block' num2str(iRun) '-final-151-1.txt'], iRun, 'T', 'stimuli');
    %     %     treeStim   = csvread([location '/treeStim-' num2str(iRun) '.csv'], 1, 0); 
    %     %     vals       = tsDifficulty(treeStim, 2);
    %     % 
    %     %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod.name  = 'treeDifficulty';
    %     %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod.param = vals - mean(vals);
    %     %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod.poly  = 1;
    %     % 
    %     % else
    %     % 
    %     %     matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod       = struct('name', {}, 'param', {}, 'poly', {});
    %     % 
    %     % end
    % end

% Specify hrf basis, mask, and noise model for objective function.
%--------------------------------------------------------------------------
% We use the canonical hrf (as opposed to a finite impulse response, for
% example) with no temporal or dispersion derivatives. One potential
% advantage to adding derivates is to account for small variations in the
% latency/width of the hrf. However, this can complicate contrast
% estimation, so we kept it simple here.
    if task == "code"
        mask_location = sprintf("/home/zachkaras/fmri/fmri_model_data/midprocess/%s/mask.nii.gz", subID);
    elseif task == "prose"
        mask_location = sprintf("/home/zachkaras/fmri/fmri_model_data/midprocess_prose/%s/mask.nii.gz", subID);
    end

    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.fact             = struct('name', {}, 'levels', {});
    % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.bases.hrf.derivs = [1 1]; % ZK adding time and dispersion derivatives
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.volt             = 1;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.global           = 'None';
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mthresh          = 0.8;
    % matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mask             = cellstr([location '/' wrw '/anatomy/t1spgr_208sl/Mask/sbrainmask.nii']);
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mask             = cellstr(mask_location);
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.cvi              = 'wls';
    
% Run design spec.
%--------------------------------------------------------------------------

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












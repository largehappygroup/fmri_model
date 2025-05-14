function csfMRIv2_designStats_rwls(nRun, durModel, paraMod)
% First-level GLM with rWLS.
%
% FORMAT csfMRIv2_designStats_rwls(nRun, durModel, paraMod)
%
%   REQUIRED INPUT:
%       nRun
%           Number of scanning runs for this subject.
%
%       durModel
%           String specifying how to model event duration, either 'rt'
%           (reaction time) or 'full' (30s).
%
%       paraMod
%           Whether or not to estimate a model with parametric modulators.
%           1 for YES, 0 for NO. This could allow us to take into account
%           the 'difficulty' of each trial.
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Create a new directory where we can store the relevant model output.
%--------------------------------------------------------------------------

    if paraMod
        unix('rm -r *rwls/paraModWorkingSize/');
        mkdir('results/glm/rwls/paraModWorkingSize/');
        cd('results/glm/rwls/paraModWorkingSize/');
        load('/Volumes/LTM/csfMRI_v2/experiment.scripts/wsDifficulty.mat');
    else 
        unix('rm -r *rwls');
        mkdir('results/glm/rwls/');
        cd('results/glm/rwls/');
    end
    
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
    
% Build design given number of runs.
%--------------------------------------------------------------------------

    for iRun = 1:nRun
    
        % Point SPM to where the data live.

        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).scans = cellstr(strcat(['../func/reviewblock/run_0' num2str(iRun) '/'], spm_select('List', ['../func/reviewblock/run_0' num2str(iRun) '/'], '^wurun.*nii$')));
    
        % Mental.
        
        mentalOnset = csvread(['../mentalOnset-' num2str(iRun) '.csv'], 1, 0);
        
        switch durModel
            case 'rt'
                mentalRT                = csvread(['../mentalRT-' num2str(iRun) '.csv'], 1, 0);
                mentalRT(mentalRT == 0) = 30;
            case 'full'
                mentalRT = 30;
        end

        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).name     = 'Mental';
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).onset    = mentalOnset;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).duration = mentalRT;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).tmod     = 0;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).orth     = 1;
        
        if paraMod
            
            mentalStim = csvread(['../mentalStim-' num2str(iRun) '.csv'], 1, 0); 
            vals       = wsDifficulty(mentalStim, 2);
            
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod.name  = 'mentalDifficulty';
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod.param = vals - mean(vals);
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod.poly  = 1;
            
        else
            
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(1).pmod       = struct('name', {}, 'param', {}, 'poly', {});
            
        end
            
        % Tree.
        
        treeOnset = csvread(['../treeOnset-' num2str(iRun) '.csv'], 1, 0);
        
        switch durModel
            case 'rt'
                treeRT              = csvread(['../treeRT-' num2str(iRun) '.csv'], 1, 0);
                treeRT(treeRT == 0) = 30;
            case 'full'
                treeRT = 30;
        end
    
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).name     = 'Tree';
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).onset    = treeOnset;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).duration = treeRT;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).tmod     = 0;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).orth     = 1;
        
        if paraMod
            
            treeStim   = csvread(['../treeStim-' num2str(iRun) '.csv'], 1, 0); 
            vals       = wsDifficulty(treeStim, 2);
            
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod.name  = 'treeDifficulty';
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod.param = vals - mean(vals);
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod.poly  = 1;
            
        else
            
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(2).pmod       = struct('name', {}, 'param', {}, 'poly', {});
            
        end
        
        % List.
        
        listOnset = csvread(['../listOnset-' num2str(iRun) '.csv'], 1, 0);
        
        switch durModel
            case 'rt'
                listRT              = csvread(['../listRT-' num2str(iRun) '.csv'], 1, 0);
                listRT(listRT == 0) = 30;
            case 'full'
                listRT = 30;
        end
    
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).name     = 'List';
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).onset    = listOnset;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).duration = listRT;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).tmod     = 0;
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).orth     = 1;
        
        if paraMod
            
            listStim   = csvread(['../listStim-' num2str(iRun) '.csv'], 1, 0); 
            vals       = wsDifficulty(listStim, 2);
            
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).pmod.name  = 'listDifficulty';
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).pmod.param = vals - mean(vals);
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).pmod.poly  = 1;
            
        else
            
            matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).cond(3).pmod       = struct('name', {}, 'param', {}, 'poly', {});
            
        end
        
        % Specify nuisance (if any) and highpass filter cutoff. If we were
        % fitting a model with AR(1) estimation of variance components,
        % we'd want to include motion parameters as nuisance variables. The
        % highpass filter just removes super-slowly fluctuating signal
        % components (< 0.01Hz) that are likely to be pure noise.
        
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).multi     = {''};
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).regress   = struct('name', {}, 'val', {});
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).multi_reg = {''};
        matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.sess(iRun).hpf       = 128;
        
    end
    
% Specify hrf basis, mask, and noise model for objective function.
%--------------------------------------------------------------------------
% We use the canonical hrf (as opposed to a finite impulse response, for
% example) with no temporal or dispersion derivatives. One potential
% advantage to adding derivates is to account for small variations in the
% latency/width of the hrf. However, this can complicate contrast
% estimation, so we kept it simple here.
    
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.fact             = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.volt             = 1;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.global           = 'None';
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mthresh          = 0.8;
    matlabbatch{1}.spm.tools.rwls.fmri_rwls_spec.mask             = cellstr('../anatomy/t1spgr*/eht*.nii');
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
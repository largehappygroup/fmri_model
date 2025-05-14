function csfMRIv2_designStats_AR1(nRun, durModel)
% First-level GLM with AR(1).
%
% FORMAT csfMRIv2_designStats_AR1(nRun, durModel)
%
%   REQUIRED INPUT:
%       nRun
%           Number of scanning runs for this subject.
%
%       durModel
%           String specifying how to model event duration, either 'rt'
%           (reaction time) or 'full' (30s).
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
    
% Create a new directory where we can store the relevant model output.
%--------------------------------------------------------------------------

    unix('rm -r *AR1');
    mkdir('results.glm.AR1');
    cd('results.glm.AR1');
    
    matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(pwd);
    
% Specify basic timing parameters.
%--------------------------------------------------------------------------

    matlabbatch{1}.spm.stats.fmri_spec.timing.units   = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT      = 0.800;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t  = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
    
% Build design given number of runs.
%--------------------------------------------------------------------------

    for iRun = 1:nRun
    
        % Point SPM to where the data live.

        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).scans = cellstr(strcat(['../func.reviewblock.run_0' num2str(iRun) '/'], spm_select('List', ['../func.reviewblock.run_0' num2str(iRun)], '^swurun.*nii$')));
    
        % Mental.
        
        mentalOnset = csvread(['../mentalOnset-' num2str(iRun) '.csv'], 1, 0);
        
        switch durModel
            case 'rt'
                mentalRT                = csvread(['../mentalRT-' num2str(iRun) '.csv'], 1, 0);
                mentalRT(mentalRT == 0) = 30;
            case 'full'
                mentalRT = 30;
        end

        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(1).name     = 'Mental';
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(1).onset    = mentalOnset;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(1).duration = mentalRT;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(1).tmod     = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(1).pmod     = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(1).orth     = 1;

        % Tree.
        
        treeOnset = csvread(['../treeOnset-' num2str(iRun) '.csv'], 1, 0);
        
        switch durModel
            case 'rt'
                treeRT              = csvread(['../treeRT-' num2str(iRun) '.csv'], 1, 0);
                treeRT(treeRT == 0) = 30;
            case 'full'
                treeRT = 30;
        end
    
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(2).name     = 'Tree';
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(2).onset    = treeOnset;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(2).duration = treeRT;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(2).tmod     = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(2).pmod     = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(2).orth     = 1;
        
        % List.
        
        listOnset = csvread(['../listOnset-' num2str(iRun) '.csv'], 1, 0);
        
        switch durModel
            case 'rt'
                listRT              = csvread(['../listRT-' num2str(iRun) '.csv'], 1, 0);
                listRT(listRT == 0) = 30;
            case 'full'
                listRT = 30;
        end
    
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(3).name     = 'List';
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(3).onset    = listOnset;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(3).duration = listRT;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(3).tmod     = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(3).pmod     = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).cond(3).orth     = 1;
        
        % Specify nuisance (if any) and highpass filter cutoff.
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).multi     = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).regress   = struct('name', {}, 'val', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).multi_reg = cellstr(strcat(['../func.reviewblock.run_0' num2str(iRun) '/'], spm_select('List', ['../func.reviewblock.run_0' num2str(iRun)], '^rp.*txt$')));
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).hpf       = 128;
        
    end
    
% Specify hrf basis, mask, and noise model for objective function.
%--------------------------------------------------------------------------
    
    matlabbatch{1}.spm.stats.fmri_spec.fact             = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt             = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global           = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh          = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask             = cellstr('../anatomy.spgrMask/sbrainmask.nii');
    matlabbatch{1}.spm.stats.fmri_spec.cvi              = 'AR(1)';
    
% Run design spec.
%--------------------------------------------------------------------------

    spm_jobman('run', matlabbatch);
    
% Run model estimation.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};

    matlabbatch{1}.spm.stats.fmri_est.spmmat           = cellstr([pwd '/SPM.mat']);
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    
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
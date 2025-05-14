function csfMRIv2_designStats_AR1(nRun, durModel, subID)
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
    location = pwd;
    wr = dir('wrw*');
    wrw = wr(1).name; 
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

        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).scans = cellstr(strcat([location '/' wrw '/func/typing/run_0' num2str(iRun) '/'], spm_select('List', [location '/' wrw '/func/typing/run_0' num2str(iRun) '/'], '^swurun*.*nii$')));
    
        % Mental.
        
        
        mentalOnset = generate_matrix([location '/codesynth-block' num2str(iRun) '-final-' subID '-1.txt'], 'M', 'onset');
        
        switch durModel
            case 'rt'
                
                mentalRT                = generate_matrix([location '/codesynth-block' num2str(iRun) '-final-' subID '-1.txt'], 'M', 'RT');
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
        
        
        treeOnset = generate_matrix([location '/codesynth-block' num2str(iRun) '-final-' subID '-1.txt'], 'T', 'onset');
        
        switch durModel
            case 'rt'
                
                treeRT              = generate_matrix([location '/codesynth-block' num2str(iRun) '-final-' subID '-1.txt'], 'T', 'RT');
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
        
        
        listOnset = generate_matrix([location '/codesynth-block' num2str(iRun) '-final-' subID '-1.txt'], 'L', 'onset');
        
        switch durModel
            case 'rt'
                
                listRT              = generate_matrix([location '/codesynth-block' num2str(iRun) '-final-' subID '-1.txt'], 'L', 'RT');
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
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).multi_reg = cellstr(strcat([location '/' wrw '/func/typing/run_0' num2str(iRun) '/'], spm_select('List', [location '/' wrw '/func/typing/run_0' num2str(iRun)], '^rp_run*.*txt$')));
        matlabbatch{1}.spm.stats.fmri_spec.sess(iRun).hpf       = 128;
        
    end
    
% Specify hrf basis, mask, and noise model for objective function.
%--------------------------------------------------------------------------
    
    matlabbatch{1}.spm.stats.fmri_spec.fact             = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt             = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global           = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh          = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask             = cellstr(strcat([location '/' wrw '/anatomy/t1spgr_208sl/Mask/sbrainmask.nii']));
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
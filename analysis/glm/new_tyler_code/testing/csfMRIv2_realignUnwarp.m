function csfMRIv2_realignUnwarp(nRun)
% Motion correction and unwarping.
%
% FORMAT csfMRIv2_realignUnwarp
%
% REQUIRED INPUT:
%   nRun
%       Number of runs completed by this subject.
%__________________________________________________________________________

% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};
        
% Tell SPM where the scans live.
%--------------------------------------------------------------------------

    % Resting-state scans first.
    
        matlabbatch{1}.spm.spatial.realignunwarp.data(1).scans  = cellstr(strcat([pwd '/func/rest/run_01/'], ...
                                                                    spm_select('List', [pwd '/func/rest/run_01/'], '^run*.*nii$')));
        matlabbatch{1}.spm.spatial.realignunwarp.data(1).pmscan = cellstr(strcat([pwd '/func/rest/run_01/'], ...
                                                                    spm_select('List', [pwd '/func/rest/run_01/'], '^vdm5_fpm0000.*img$')));
        
    % Now grab the remaining EPIs.

    for iRun = 1:nRun
                
        matlabbatch{1}.spm.spatial.realignunwarp.data(iRun+1).scans  = cellstr(strcat([pwd '/func/reviewblock/run_0' num2str(iRun) '/'], ...
                                                                        spm_select('List', [pwd '/func/reviewblock/run_0' num2str(iRun) '/'], '^run*.*nii$')));
        matlabbatch{1}.spm.spatial.realignunwarp.data(iRun+1).pmscan = cellstr(strcat([pwd '/func/reviewblock/run_0' num2str(iRun) '/'], ...
                                                                        spm_select('List', [pwd '/func/reviewblock/run_0' num2str(iRun) '/'], '^vdm5_fpm0000.*img$')));
        
    end
    
% Define all parameters for realign/unwarp estimation.
%--------------------------------------------------------------------------

    % Motion estimation for realignment - this is where we compute our 6
    % motion parameters (translations along x,y,z and rotations around
    % x,y,z). We don't change much from the SPM defaults: the main
    % difference is that we set our interpolant to a 7th degree B-spline
    % (this increases the computation time but it provides a much more
    % robust estimate).

        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.quality = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.sep     = 4;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.fwhm    = 5;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.rtm     = 0;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.einterp = 7;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.ewrap   = [0 0 0];
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.weight  = '';

    % Deformation field estimation for unwarping - here we're dealing with
    % the fact that head motion interacts with the baseline distortion of
    % the scanner, causing all sorts of nasty field inhomogeneities. The
    % result is that our ideally cubic voxels get geometrically warped. We
    % take advantage of the fieldmaps we collected to get a good idea of
    % the distribution of the magnetic field and how motion might distort
    % it further.

        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.basfcn   = [12 12];
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.lambda   = 100000;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.jm       = 0;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.fot      = [4 5];
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.sot      = [];
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.uwfwhm   = 4;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.rem      = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.noi      = 5;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';

    % Interpolation/reslicing - once we have all the factors above
    % estimated, we need to 'reslice' the data back into the same space
    % (essentially creating a new grid of voxels based on how things
    % might've moved around or been geometrically-skewed).

        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.rinterp = 7;
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.wrap    = [0 0 0];
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.mask    = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.prefix  = 'u';

% Run motion/distortion correction.
%--------------------------------------------------------------------------

    spm_jobman('run', matlabbatch);

% Create new folder for the mean functional image and move it there.
%--------------------------------------------------------------------------

    mkdir('./func/mean');
    unix('mv ./func/rest/run_01/meanurun* ./func/mean');

% Estimate framewise displacement for each scanning run.
%--------------------------------------------------------------------------
% Here, we convert rotational displacements to translations; take the first 
% derivative (TR-to-TR movement); and sum across columns (total movement 
% between each frame). We won't necessarily use this in any subsequent 
% analyses but it's a good 'quality assurance' measure that summarizes how
% much a subject moved during each scan.

    % Initialize structure for resting-state.

        framewiseDisplacement.rest = [];
        
    % Obtain realignment parameters.    
    
        rpData = load(strcat([pwd '/func/rest/run_01/'], ...
                        spm_select('List', [pwd '/func/rest/run_01/'], 'rp_tprun_01.txt')));
        
    % Compute and store. 
        
        rpData(:,4:6) = rpData(:,4:6) .* 50;
        dx            = diff(rpData);
        fwd           = sum(abs(dx),2);
            
        framewiseDisplacement.rest.series = fwd;
        framewiseDisplacement.rest.mean   = mean(fwd);
        framewiseDisplacement.rest.max    = max(fwd);

    % Loop over the remaining EPIs.

        for iRun = 1:nRun
        
            % Initialize structure for this run.

                framewiseDisplacement.(['run_0' num2str(iRun)]) = [];
        
            % Obtain realignment parameters.    
    
                rpData = load(strcat([pwd '/func/reviewblock/run_0' num2str(iRun) '/'], ...
                                spm_select('List', [pwd '/func/reviewblock/run_0' num2str(iRun) '/'], ['rp_tprun_0' num2str(iRun) '.txt'])));
        
            % Compute and store. 
        
                rpData(:,4:6) = rpData(:,4:6) .* 50;
                dx            = diff(rpData);
                fwd           = sum(abs(dx),2);
            
                framewiseDisplacement.(['run_0' num2str(iRun)]).series = fwd;
                framewiseDisplacement.(['run_0' num2str(iRun)]).mean   = mean(fwd);
                framewiseDisplacement.(['run_0' num2str(iRun)]).max    = max(fwd);
            
        end
            
    % Save structure.
        
        mkdir('./fwd');
        cd('./fwd');
        save framewiseDisplacement framewiseDisplacement;
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
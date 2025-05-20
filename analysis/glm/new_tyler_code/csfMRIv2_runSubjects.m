function csfMRIv2_runSubjects(whichProc)
% Wrapper script to iterate over subjects and perform preprocessing /
% first-level analysis.
%
% FORMAT csfMRIv2_runSubjects(whichProc)
%
% REQUIRED INPUT:
%   whichProc
%       String specifying which process(es) to execute, either 'preprocess'
%       or 'task' (for first-level GLM).
%__________________________________________________________________________

% Load subject list and run counts.
%--------------------------------------------------------------------------

    %load('/Volumes/LTM/csfMRI_v2/experiment.scripts/allSubjects.mat');
    %load('/Volumes/LTM/csfMRI_v2/experiment.scripts/nRuns.mat');
	load('allSubjects.mat');
	load('nRuns.mat');
    nSub = length(allSubjects);
    
% Start the clock so we can track overall computation time.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

    procStart = tic;
    
% Begin looping over subjects.
%--------------------------------------------------------------------------
    
    disp(' ');
    
    parentDir = pwd;
    
    for iSub = 1:nSub
        % here, you specify and bad data that you want excluded form the anlaysis
        if ((nRuns(iSub) == 0) || (str2double(allSubjects{iSub}) == 160) || (str2double(allSubjects{iSub}) == 163) || (str2double(allSubjects{iSub}) == 203))
            
            continue
            
        else
        
            subID = allSubjects{iSub};
        
            disp(['- Running subject: ' subID '. Please wait... -']);
        
            cd(subID);
        
            %unix('rm -r ./spm*');
        
            switch whichProc
            
                case 'preprocess'
            
                    % The functions below will run through a standard 
                    % preprocessing pipeline for fMRI analysis.
            
                        csfMRIv2_realignUnwarp(nRuns(iSub));
                        csfMRIv2_segmentStrip('spgr');
                        %csfMRIv2_segmentStrip('flair');
                        csfMRIv2_brainMask('spgr', 1);
                        %csfMRIv2_brainMask('flair', 1);
                        csfMRIv2_coregister(nRuns(iSub), 'spgr');
                        csfMRIv2_normalise(nRuns(iSub), 'spgr');
                        csfMRIv2_smooth(nRuns(iSub));
                    
                case 'task'
                    
                    % Specify and estimate first-level GLM, then compute
                    % contrasts.
                    
                        %csfMRIv2_designStats_AR1(nRuns(iSub), 'rt');
                        %csfMRIv2_addContrasts(nRuns(iSub), 'AR1');
                    
                        csfMRIv2_designStats_rwls(nRuns(iSub), 'rt', 1, subID);
                        % csfMRIv2_addContrasts(nRuns(iSub), 'rwls');
                        %csfMRIv3_smoothCons(subID, whichTask);
                case 'smooth'
                    %In the case of 2nd-level analysis, the funtions below
                    %will prepare our contrast images
                        csfMRIv3_smoothCons(subID, 'rwls');
                
            end
            
            cd(parentDir);
            
        end
        
    end
    
% Display total computation time.
%--------------------------------------------------------------------------

    procEnd = toc(procStart);
    disp(['- Jobs completed for ' num2str(nSub) ' subjects in ' num2str(procEnd/60) ' minutes -']);
    
end
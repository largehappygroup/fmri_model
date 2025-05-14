function designStats_AR1()
	load('allSubjects.mat');
	load('nRuns.mat');
    nSub = length(allSubjects);
    procStart = tic;    
    disp(' ');
    parentDir = pwd;
    
    for iSub = 20:nSub
        
        if ((nRuns(iSub) == 0) || (str2double(allSubjects{iSub}) == 160) || (str2double(allSubjects{iSub}) == 163) || (str2double(allSubjects{iSub}) == 203))
            
            continue
            
        else
        
            subID = allSubjects{iSub};
        
            disp(['- Running subject: ' subID '. Please wait... -']);
        
            cd(subID);
            csfMRIv2_designStats_AR1(nRuns(iSub), 'rt', subID); 
            
            cd(parentDir);
            
        end
        
    end
    
% Display total computation time.
%--------------------------------------------------------------------------

    procEnd = toc(procStart);
    disp(['- Jobs completed for ' num2str(nSub) ' subjects in ' num2str(procEnd/60) ' minutes -']);
    
end
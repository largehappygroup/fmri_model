function csfMRIv2_addContrasts(nRun, estMethod)
% Construct task-related contrasts.
%
% FORMAT csfMRIv2_addContrasts(nRun, estMethod)
%
%   REQUIRED INPUT:
%       nRun
%           Number of scanning runs for this subject.
%
%       estMethod
%           String specifying GLM estimation method, either 'AR1' or
%           'rwls'.
%__________________________________________________________________________


% Initialize default SPM configurations for fMRI.
%--------------------------------------------------------------------------

    setDefaultsSPM;
    
    matlabbatch = {};

% Navigate into the results directory for this task.
%--------------------------------------------------------------------------
	
	cd(['results.glm.' estMethod '.paraModWorkingSize']);
		
% Load the design matrix.
%--------------------------------------------------------------------------

    load([pwd '/SPM.mat']);
		
% Define conditions for this task.
%--------------------------------------------------------------------------
% Note that the rWLS models included parametric modulators for trial 
% difficulty.
	
    switch estMethod
        case 'AR1'
            conditions{1} = 'Mental';
            conditions{2} = 'Tree';
            conditions{3} = 'List';
        case 'rwls'
            conditions{1} = 'Mental';
            conditions{2} = 'mentalDifficulty';
            conditions{3} = 'Tree';
            conditions{4} = 'treeDifficulty';
            conditions{5} = 'List';
            conditions{6} = 'listDifficulty';
    end
		
% Specify relevant contrasts.
%--------------------------------------------------------------------------
% A contrast vector is essentially a bunch of 0s, 1s, and -1s that
% determine which conditions / event types we want to compare. Accordingly,
% contrasts are directional - we're looking for voxels where activity in
% one condition is significantly greater than another (it's essentially a
% one-tailed test). For nuisance variables (e.g. motion parameters) or
% conditions we don't want to factor into our analysis, we simply specify
% zeros. Keep in mind that the contrast vector needs to sum to zero for a
% 'true' task related comparison (but there are cases where it might not,
% e.g. if we want to assess brain activity vs. an implicit baseline).

	contrasts{1}.name = 'Task > Base';
    switch estMethod
        case 'AR1'
            contrasts{1}.vector = repmat([1 1 1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{1}.vector = repmat([1 0 1 0 1 0], 1, nRun);
    end
		
	contrasts{2}.name = 'Mental > Tree';
    switch estMethod
        case 'AR1'
            contrasts{2}.vector = repmat([1 -1 0 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{2}.vector = repmat([1 0 -1 0 0 0], 1, nRun);
    end
		
	contrasts{3}.name = 'Tree > Mental';
    switch estMethod
        case 'AR1'
            contrasts{3}.vector = repmat([-1 1 0 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{3}.vector = repmat([-1 0 1 0 0 0], 1, nRun);
    end
		
	contrasts{4}.name = 'Mental > List';
    switch estMethod
        case 'AR1'
            contrasts{4}.vector = repmat([1 0 -1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{4}.vector = repmat([1 0 0 0 -1 0], 1, nRun);
    end
    
    contrasts{5}.name = 'List > Mental';
    switch estMethod
        case 'AR1'
            contrasts{5}.vector = repmat([-1 0 1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{5}.vector = repmat([-1 0 0 0 1 0], 1, nRun);
    end
    
    contrasts{6}.name = 'Tree > List';
    switch estMethod
        case 'AR1'
            contrasts{6}.vector = repmat([0 1 -1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{6}.vector = repmat([0 0 1 0 -1 0], 1, nRun);
    end
    
    contrasts{7}.name = 'List > Tree';
    switch estMethod
        case 'AR1'
            contrasts{7}.vector = repmat([0 -1 1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{7}.vector = repmat([0 0 -1 0 1 0], 1, nRun);
    end
    
    contrasts{8}.name = 'Mental > Base';
    switch estMethod
        case 'AR1'
            contrasts{8}.vector = repmat([1 0 0 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{8}.vector = repmat([1 0 0 0 0 0], 1, nRun);
    end
    
    contrasts{9}.name = 'Tree > Base';
    switch estMethod
        case 'AR1'
            contrasts{9}.vector = repmat([0 1 0 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{9}.vector = repmat([0 0 1 0 0 0], 1, nRun);
    end
    
    contrasts{10}.name = 'List > Base';
    switch estMethod
        case 'AR1'
            contrasts{10}.vector = repmat([0 0 1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{10}.vector = repmat([0 0 0 0 1 0], 1, nRun);
    end
    
    contrasts{11}.name = 'Mental > Code';
    switch estMethod
        case 'AR1'
            contrasts{11}.vector = repmat([1 -1 -1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{11}.vector = repmat([1 0 -1 0 -1 0], 1, nRun);
    end
    
    contrasts{12}.name = 'Code > Mental';
    switch estMethod
        case 'AR1'
            contrasts{12}.vector = repmat([-1 1 1 zeros(1,6)], 1, nRun);
        case 'rwls'
            contrasts{12}.vector = repmat([-1 0 1 0 1 0], 1, nRun);
    end
    
    switch estMethod
        
        case 'rwls'
            
            contrasts{13}.name   = 'Mental > Tree (Linear)';
            contrasts{13}.vector = repmat([0 1 0 -1 0 0], 1, nRun);
            
            contrasts{14}.name   = 'Tree > Mental (Linear)';
            contrasts{14}.vector = repmat([0 -1 0 1 0 0], 1, nRun);
            
            contrasts{15}.name   = 'Mental > List (Linear)';
            contrasts{15}.vector = repmat([0 1 0 0 0 -1], 1, nRun);
            
            contrasts{16}.name   = 'List > Mental (Linear)';
            contrasts{16}.vector = repmat([0 -1 0 0 0 1], 1, nRun);
            
            contrasts{17}.name   = 'Tree > List (Linear)';
            contrasts{17}.vector = repmat([0 0 0 1 0 -1], 1, nRun);
            
            contrasts{18}.name   = 'List > Tree (Linear)';
            contrasts{18}.vector = repmat([0 0 0 -1 0 1], 1, nRun);
            
            contrasts{19}.name   = 'Mental > Base (Linear)';
            contrasts{19}.vector = repmat([0 1 0 0 0 0], 1, nRun);
            
            contrasts{20}.name   = 'Tree > Base (Linear)';
            contrasts{20}.vector = repmat([0 0 0 1 0 0], 1, nRun);
            
            contrasts{21}.name   = 'List > Base (Linear)';
            contrasts{21}.vector = repmat([0 0 0 0 0 1], 1, nRun);
            
            contrasts{22}.name   = 'Mental > Code (Linear)';
            contrasts{22}.vector = repmat([0 1 0 -1 0 -1], 1, nRun);
            
            contrasts{23}.name   = 'Code > Mental (Linear)';
            contrasts{23}.vector = repmat([0 -1 0 1 0 1], 1, nRun);
                        
    end
        
	% Create actual contrast vectors with balancing/weighting for multiple
	% scanning runs.
	
		nConditions = length(conditions);
		nPredictors = size(SPM.xX.iC,2);
		
		disp(' ');
	
		for iContrast = 1:length(contrasts)
		
			% Setup input and output.
            
			inputVect  = contrasts{iContrast}.vector;		
			outputVect = zeros(1,nPredictors);
		
			% Cycle through the predictors to ID the conditions.
            
			for jCondition = 1:nConditions
		
				conditionTag = conditions{jCondition};
				
				matchCount = 0;
				
				for kPredictor = 1:nPredictors
				
					predictorTag = SPM.xX.name{kPredictor};
					
					findResult   = regexp(predictorTag,conditionTag);

					if (findResult > 0)
						outputVect(kPredictor) = 1 * inputVect(jCondition);
						matchCount             = matchCount + 1;
					end
				
				end
				
				if ((matchCount == 0) && (inputVect(jCondition) ~= 0))
					disp(['     ERROR: ''' conditionTag ''' condition not matched in ' contrasts{iContrast}.name]);
				end

			end

			% Balance the values if necessary.
            
			if (any(outputVect == -1))
			
				positiveCount = sum(double(outputVect == 1));
				negativeCount = sum(double(outputVect == -1));
				
				outputVect(outputVect == 1)  = 1 / positiveCount;
				outputVect(outputVect == -1) = -1 / negativeCount;
				
			end
			
			contrasts{iContrast}.output = outputVect;
		
		end
		
		disp(' ');


	% Generate and run the SPM contrast job.
	
		matlabbatch{1}.spm.stats.con.spmmat = cellstr([pwd '/SPM.mat']);
        matlabbatch{1}.spm.stats.con.delete = 1;
	
		for iContrast = 1:length(contrasts)
		
			matlabbatch{1}.spm.stats.con.consess{iContrast}.tcon.name    = contrasts{iContrast}.name;
			matlabbatch{1}.spm.stats.con.consess{iContrast}.tcon.convec  = contrasts{iContrast}.output;
			matlabbatch{1}.spm.stats.con.consess{iContrast}.tcon.sessrep = 'none';
		
		end
		
		spm_jobman('run',matlabbatch);
		
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
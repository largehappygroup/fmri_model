% if  ~exist('./data/nsdflocexampledataset.mat','file')
%     % download data with curl
%     system('curl -L --output ./data/nsdflocexampledataset.mat https://osf.io/g42tm/download')
% end

% load('/home/zachkaras/fmri/fmri_model/analysis/glm/data/nsdflocexampledataset.mat')
% calculating single trial betas using GLMsingle

% output directories
outputdir_root  = "/home/zachkaras/fmri/fmri_model_data/beta_maps";
code_outputdir  = sprintf("%s/code/conditions", outputdir_root);
prose_outputdir = sprintf("%s/prose/conditions", outputdir_root);

tr = 0.8;
stimdur = 60;

% iterate_through_participants(prose_datapath, "prose")
code_datapath = "/home/zachkaras/fmri/fmri_model_data/midprocess/";
iterate_through_participants(code_datapath, "code", code_outputdir, stimdur, tr)

prose_datapath = "/home/zachkaras/fmri/fmri_model_data/midprocess_prose/";
iterate_through_participants(porse_datapath, "prose", prose_outputdir, stimdur, tr)


% need to load in corresponding brain data
function iterate_through_participants(datapath, task, outputdir, stimdur, tr)
    
    files = dir(datapath); 
    for i=3:numel(files) 
        if ~contains(files(i).name, 'pkl')
            fnames{i-2}=files(i).name; 
        end
    end

    for i=3:length(fnames)
        person = fnames{i};
        brain_data_path = sprintf("%s/%s/filtered_func_data_clean.nii.gz", datapath, person);
        brain_data = niftiread(brain_data_path);
        design = create_design_matrix(fnames{i}, task);

        opt = struct('wantmemoryoutputs',[1 1 1 1]);

        % This example saves output .mat files to the folder
        % "example2outputs/GLMsingle". If these outputs don't already exist, we
        % will perform the time-consuming call to GLMestimatesingletrial.m;
        % otherwise, we will just load from disk.
        full_outputdir = sprintf('%s/%s/GLMsingle', outputdir, person);
        if ~exist(full_outputdir,'dir')
        
            [results] = GLMestimatesingletrial(design,brain_data,stimdur,tr,full_outputdir,opt);
        
            % We assign outputs of GLMestimatesingletrial to "models" structure.
            % Note that results{1} contains GLM estimates from an ONOFF model,
            % where all images are treated as the same condition. These estimates
            % could be potentially used to find cortical areas that respond to
            % visual stimuli. We want to compare beta weights between conditions
            % therefore we are not going to store the ONOFF GLM results.
        
            clear models;
            models.FIT_HRF = results{2};
            models.FIT_HRF_GLMdenoise = results{3};
            models.FIT_HRF_GLMdenoise_RR = results{4};
        
        else
            % Load existing file outputs if they exist
            results = load([outputdir '/GLMsingle/TYPEB_FITHRF.mat']);
            models.FIT_HRF = results;
            results = load([outputdir '/GLMsingle/TYPEC_FITHRF_GLMDENOISE.mat']);
            models.FIT_HRF_GLMdenoise = results;
            results = load([outputdir '/GLMsingle/TYPED_FITHRF_GLMDENOISE_RR.mat']);
            models.FIT_HRF_GLMdenoise_RR = results;
        end
        break

    end
end


function X = create_question_design_matrix(person)
    % if the task is code, load regressors from loops and nonloops
    % if the task is prose, load regressors from prose/choice and and prose/explain_both 
    base_datapath = "/home/zachkaras/fmri/fmri_model/midprocessing/regressors/questions";

    X = [];
    for i=0:8
        regressor_path = sprintf("%s/%d/%s.csv", base_datapath, i, person);
        try
            regressor = readtable(regressor_path);
        catch
            fprintf("Couldn't find file for %s on question %d\n", person, i)
            continue
        end
        
        X = [X, table2array(regressor)];
    end
end

function X = create_design_matrix(person, task)
    % if the task is code, load regressors from loops and nonloops
    % if the task is prose, load regressors from prose/choice and and prose/explain_both 
    base_datapath = "/home/zachkaras/fmri/fmri_model/midprocessing/regressors";

    if strcmp(task, 'code')
        cond_one_path = sprintf("%s/loops_nonloops/loops/%s.csv", base_datapath, person);
        cond_two_path = sprintf("%s/loops_nonloops/nonloops/%s.csv", base_datapath, person);
    elseif strcmp(task, 'prose')
        cond_one_path = sprintf("%s/prose/choice/%s.csv", base_datapath, person);
        cond_two_path = sprintf("%s/prose/explain_both/%s.csv", base_datapath, person);
    end

    cond_one_onsets = readtable(cond_one_path);
    cond_two_onsets = readtable(cond_two_path);
    X = [table2array(cond_one_onsets), table2array(cond_two_onsets)];
end





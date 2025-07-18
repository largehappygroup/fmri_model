prose_datapath = "/home/zachkaras/fmri/fmri_model_data/clean_prose";
code_datapath = "/home/zachkaras/fmri/fmri_model_data/clean";

% iterate_through_participants(prose_datapath, "prose")
iterate_through_participants(code_datapath, "code")


function iterate_through_participants(datapath, task)

    files = dir(datapath); for i=3:numel(files); fnames{i-2}=files(i).name; end
    if task == "prose"
        blocknum = 1;
    elseif task == "code"
        blocknum = 3;
    end

    for i=3:length(files)
        person = files(i).name(1:3);

        outpath_question = sprintf("/home/zachkaras/fmri/fmri_model_data/beta_maps/%s/questions/%s", task, person);

        if ~exist(outpath_question, 'dir')
            mkdir(outpath_question)
        else
            fprintf("Skipping %s, already completed.\n", person)
            continue
        end

        fprintf("\nRunning GLMs for participant %s\n", person)
        
        % The data in the 'clean' directories may have been overprocessed
        if task == "prose"
            fmri_datapath = sprintf("/home/zachkaras/fmri/fmri_model_data/midprocess_prose/%s/filtered_func_data_clean.nii.gz", person);
        elseif task == "code"
            fmri_datapath = sprintf("/home/zachkaras/fmri/fmri_model_data/midprocess/%s/filtered_func_data_clean.nii.gz", person);
        end
        
        uncompress_file = sprintf("gunzip %s", fmri_datapath);
        system(uncompress_file);
        uncompressed_datapath = char(fmri_datapath);
        uncompressed_datapath = uncompressed_datapath(1:end-3);
        
        if exist(uncompressed_datapath) < 1
            continue
        end

        % Each question individually (question onset file)
        disp("Running GLM for each question")

        
        question_onsets = create_design_matrix(person);
        if isempty(question_onsets)
            fprintf("Issue with data from participant %s, skipping.", person)
        end

        if (exist(sprintf("%s/SPM.mat", outpath_question)) < 1) && (size(question_onsets,2) == 9)
            try
                SPM_GLM_single_trial(person, uncompressed_datapath, outpath_question, question_onsets, task)
            catch e
                fprintf("Issue with data from participant %s: %s\n", person, e.message)
                getReport(e)
            end
        end

        compress_file = sprintf("gzip %s", uncompressed_datapath);
        system(compress_file);
        % break
    end
end


function X = create_design_matrix(person)
    base_datapath = "/home/zachkaras/fmri/fmri_model/midprocessing/regressors/questions";

    X = [];
    for i=0:8
        regressor_path = sprintf("%s/%d/%s.csv", base_datapath, i, person);
        
        try
            regressor = readtable(regressor_path);
        catch
            X = [];
            return
        end
        
        try
            X = [X, table2array(regressor)];
        catch
            X = [];
            return
        end
    end
end



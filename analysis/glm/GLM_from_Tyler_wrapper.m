%% Performing GLM using SPM code

% Load all participants
% Even though I'm getting the filenames from the clean and clean_prose
% directories, those data have been preprocessed too much since I applied
% nuisance regression after ICA 
prose_datapath = "/home/zachkaras/fmri/fmri_model_data/clean_prose";
code_datapath = "/home/zachkaras/fmri/fmri_model_data/clean";

iterate_through_participants(prose_datapath, "prose")
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
        fprintf("\nRunning GLMs for participant %s\n", person)
        % fmri_datapath = sprintf("%s/%s", datapath, files(i).name);
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
        
        % run GLM separately for the following:
        % Specify an output folder for each as well

        % All questions (full onset file)
        fprintf("Running GLM for all questions\n")
        outpath_all = sprintf("/home/zachkaras/fmri/fmri_model_data/beta_maps/%s/all/%s", task, person);
        
        onset_all = sprintf("/home/zachkaras/fmri/fmri_model/data/%s/relative-onsets-%s-%d.txt", person, person, blocknum);
        condition_all = sprintf("%s_all_questions", task);
        
        if (exist(sprintf("%s/SPM.mat", outpath_all)) < 1) && (exist(onset_all) > 0)
            try
                SPM_GLM(person, condition_all, uncompressed_datapath, outpath_all, onset_all, task)
            catch
                fprintf("Issue with data from participant %s on all questions\n", person)
            end
        end

        % Each question individually (question onset file)
        fprintf("Running GLM for question...")
        for ii=0:8
            fprintf("%d...", ii)
            outpath_question = sprintf("/home/zachkaras/fmri/fmri_model_data/beta_maps/%s/questions/%d/%s", task, ii, person);
            onset_question = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/onsets/%s/questions/%d/%s.csv", task, ii, person);
            condition_question = sprintf("%s_question_%d", task, ii);
            if (exist(sprintf("%s/SPM.mat", outpath_question)) < 1) && (exist(onset_question) > 0)
                try
                    SPM_GLM(person, condition_question, uncompressed_datapath, outpath_question, onset_question, task)
                catch
                    fprintf("Issue with data from participant %s on question %d\n", person, ii)
                end
            end
        end

        % Loops/nonloops for code
        if task == "code"
            fprintf("\nRunning GLM for loops/nonloops\n")
            outpath_loops = sprintf("/home/zachkaras/fmri/fmri_model_data/beta_maps/code/loops/%s", person);
            onset_loops = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/onsets/code/loops/%s.csv", person);
            condition_loops = "code_loops";
            if (exist(sprintf("%s/SPM.mat", outpath_loops)) < 1) && (exist(onset_loops) > 0)
                try
                    SPM_GLM(person, condition_loops, uncompressed_datapath, outpath_loops, onset_loops, task)
                catch
                    fprintf("Issue with data from participant %s on loops\n", person)
                end
            end

            outpath_nonloops = sprintf("/home/zachkaras/fmri/fmri_model_data/beta_maps/code/nonloops/%s", person);
            onset_nonloops = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/onsets/code/nonloops/%s.csv", person);
            condition_nonloops = "code_nonloops";
            if (exist(sprintf("%s/SPM.mat", outpath_nonloops)) < 1) && (exist(onset_nonloops) > 0)
                try
                    SPM_GLM(person, condition_nonloops, uncompressed_datapath, outpath_nonloops, onset_nonloops, task)
                catch
                    fprintf("Issue with data from participant %s on nonloops\n", nonperson)
                end
            end
        end

        compress_file = sprintf("gzip %s", uncompressed_datapath);
        system(compress_file);
        % break
    end
end




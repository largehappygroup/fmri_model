%% Perform quality control checks based on motion artifacts in the data

% Two directories containing participants' fMRI data and motion recordings
code_datadir  = "/home/zachkaras/fmri/fmri_model_data/midprocess";
prose_datadir = "/home/zachkaras/fmri/fmri_model_data/midprocess_prose";

% Calculating metrics related to noise for all participants
code_outputs = loop_through_participants(code_datadir);
prose_outputs = loop_through_participants(prose_datadir);

% Checking and plotting measures associated with measures of mean framewise
% displacement and overall percentage of volumes that exceeded the motion
% threshold
print_noise_metrics(code_outputs, "Code")
print_noise_metrics(prose_outputs, "Prose")

function print_noise_metrics(noise_struct, taskname)
    percentages = [];
    mean_fd = [];
    ids = {};

    % Collecting mean, overall percentage, and ids for participants
    for i=1:numel(noise_struct)
        ids{end+1} = noise_struct(i).id;
        percentages(end+1) = noise_struct(i).noise_measures.spikePercentage;
        mean_fd(end+1) = noise_struct(i).noise_measures.meanFD;
    end
    
    figure;
    subplot(2, 1, 1)
    plot(percentages)
    xticks(1:numel(ids))
    xticklabels(ids)
    title(sprintf("%s Percentages", taskname))
    
    subplot(2,1,2)
    plot(mean_fd)
    xticks(1:numel(ids))
    xticklabels(ids)
    title(sprintf("%s Mean Displacement", taskname))
end


function qa_outputs = loop_through_participants(datadir)
    files = dir(datadir); 
    
    % getting participant file names
    for i=3:numel(files) 
        if files(i).isdir 
            fnames{i-2}=files(i).name; 
        end 
    end

    % Empty struct that will hold measures for each participant
    qa_outputs = struct('id',{},'noise_measures',{});
    for i=1:length(fnames)
        
        motionParams_file = sprintf("%s/%s/mc/prefiltered_func_data_mcf.par", datadir, fnames{i});
        try
            motionParams = load(motionParams_file);
        catch
            fprintf("Can't find motion parameters for participant %s\n", fnames{i})
            continue
        end
        % Running Tyler's script for calculating QC measures
        qa_outputs(end+1) = struct('id', {fnames{i}}, 'noise_measures', {motionQA(motionParams)});
        
    end
end

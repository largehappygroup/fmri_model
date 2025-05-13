%%
% make_task_regressor(filename, nframes, TR, dur, onsets)
% read in all the 'processed-answers' files
datapath = "/home/zachkaras/fmri/fmri_model_data/midprocess";
contents = dir(datapath);

contents = contents([contents.isdir]);
participants = contents(~ismember({contents.name}, {'.', '..'}));

% get order of conditions
for i=1:length(participants)
    
    person = participants(i).name;
    disp(person)

    %% Reading in fMRI file to get number of frames
    brain_file = sprintf("/home/zachkaras/fmri/fmri_model_data/clean/%s", person);
    try
        brain_data = niftiinfo(brain_file);
    catch
        fprintf("No fMRI data for %s", person)
        continue
    end
    nframes = brain_data.ImageSize(4);
    % nframes = 746;
    
    %% Reading in Task Info
    onset_file = sprintf("/home/zachkaras/fmri/fmri_model/data/%s/relative-onsets-%s-3.txt", person, person);
    task_info = readtable(onset_file, 'Delimiter', ' ', 'ReadVariableNames', false);
    
    stim_ids = task_info.Var1;
    onsets = task_info.Var2;
    
    create_question_regressors(person, nframes, stim_ids, onsets);
    create_loop_regressors(person, nframes, stim_ids, onsets);
 
    % break
end

% find volumes associated with loops, nonloops


% find volumes associated with each condition


% create file names


% save in proper directories

%% Function to create regressors for each question
% Doesn't return anything, just saves to file
function create_question_regressors(person, nframes, stim_ids, onsets)
    % disp(onsets)
    % dur = stimuli.duration;
    dur = 60;
    TR = 0.8;
    dur_in_vols = dur/TR;
    
    % read in file with volumes for the condition
    % disp(onsets)
    for i=1:length(onsets)
        task = zeros(nframes,1);
        id = stim_ids(i);
        if onsets(i) < (length(task)*TR)

            startTime = onsets(i);
            startVol = startTime/TR;
            rounded_startVol = round(startVol);
            rounded_dur_in_vols = round(dur_in_vols);

            task(rounded_startVol:rounded_startVol + rounded_dur_in_vols) = 1; % double check this
            savepath = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/questions/%d/%s.csv", id, person);
            writematrix(task, savepath);
            
        end
        % break
    end

end



%% Function to create regressors for loops/nonloops
% Doesn't return anything, just saves to file
function create_loop_regressors(person, nframes, stim_ids, onsets)
    dur = 60;
    TR = 0.8;
    dur_in_vols = dur/TR;

    % read in file with volumes for the condition
    % loops: 3, 4, 5, 6, 8
    % conditionals: 0, 1, 2, 7
    loops = [3 4 5 6 8];

    loop_task = zeros(nframes,1);
    nonloop_task = zeros(nframes,1);
    
    for i=1:length(onsets)
        id = stim_ids(i);
        if ismember(id, loops)
            category = "loops";
        else
            category = "nonloops";
        end

        sprintf("%d : %s", id, category)
        if onsets(i) < (length(loop_task)*TR)

            startTime = onsets(i);
            startVol = startTime/TR;
            rounded_startVol = round(startVol);
            rounded_dur_in_vols = round(dur_in_vols);
        
            if strcmp(category, "loops")
                loop_task(rounded_startVol:rounded_startVol + rounded_dur_in_vols) = 1; % double check this
            elseif strcmp(category, "nonloops")
                nonloop_task(rounded_startVol:rounded_startVol + rounded_dur_in_vols) = 1;
            end
        end
    end

    loop_savepath = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/loops_nonloops/loops/%s.csv", person);
    nonloop_savepath = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/loops_nonloops/nonloops/%s.csv", person);
    
    writematrix(loop_task, loop_savepath);
    writematrix(nonloop_task, nonloop_savepath);
end








%%
% make_task_regressor(filename, nframes, TR, dur, onsets)
% read in all the 'processed-answers' files
datapath = "/home/zachkaras/fmri/fmri_model_data/midprocess";
contents = dir(datapath);

contents = contents([contents.isdir]);
participants = contents(~ismember({contents.name}, {'.', '..'}));

% Long Response Prose
iterate_through_directories(participants, 1)

% Long Response Code
iterate_through_directories(participants, 3)


function iterate_through_directories(participants, blocknum)
    
    for i=1:length(participants)
        
        person = participants(i).name;
        disp(person)
        
        % Reading in task info
        onset_file = sprintf("/home/zachkaras/fmri/fmri_model/data/%s/relative-onsets-%s-%d.txt", person, person, blocknum);
        task_info = readtable(onset_file, 'Delimiter', ' ', 'ReadVariableNames', false);
        
        stim_ids = task_info.Var1;
        onsets = task_info.Var2;
        
        create_question_onsets(person, blocknum, stim_ids, onsets);
        
        if blocknum == 3
            create_loop_onsets(person, stim_ids, onsets);
        end

        % break
    end
end


%% Function to create onset files for each question
% Doesn't return anything, just saves to file
function create_question_onsets(person, blocknum, stim_ids, onsets)
    if blocknum == 1
        task = "prose";
    elseif blocknum == 3
        task = "code";
    end
   
    % read in file with volumes for the condition
    for i=1:length(onsets)     
        id = stim_ids(i);
        startTime = onsets(i);
        savepath = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/onsets/%s/questions/%d/%s.csv", task, id, person);
        writematrix(startTime, savepath);
    end
end



%% Function to create onsets for loops/nonloops
% Doesn't return anything, just saves to file
function create_loop_onsets(person, stim_ids, onsets)
    % read in file with volumes for the condition
    % loops: 3, 4, 5, 6, 8
    % conditionals: 0, 1, 2, 7
    loops = [3 4 5 6 8];

    loop_onsets = [];
    nonloop_onsets = [];
    
    for i=1:length(onsets)
        id = stim_ids(i);
        if ismember(id, loops)
            category = "loops";
        else
            category = "nonloops";
        end

        % sprintf("%d : %s", id, category)
        startTime = onsets(i);
        
        if strcmp(category, "loops")
            loop_onsets(end+1) = startTime;
        elseif strcmp(category, "nonloops")
            nonloop_onsets(end+1) = startTime;
        end
    end

    loop_savepath = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/onsets/code/loops_nonloops/loops/%s.csv", person);
    nonloop_savepath = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/onsets/code/loops_nonloops/nonloops/%s.csv", person);
    
    writematrix(loop_onsets', loop_savepath);
    writematrix(nonloop_onsets', nonloop_savepath);
end








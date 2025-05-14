function generate_csv(filename, iRun, taskName, onsetOrRT)
input = fopen(filename);
if onsetOrRT
    switch taskName
        case 'M'
            output = ['mentalOnset-' num2str(iRun) '.csv'];
            grabString = 'taskM.OnsetTime:';
        case 'L'
            output = ['listOnset-' num2str(iRun) '.csv'];
            grabString = 'taskL.OnsetTime:';
        case 'T'
            output = ['treeOnset-' num2str(iRun) '.csv'];
            grabString = 'taskT.OnsetTime:';
    end
else
    switch taskName
        case 'M'
            output = ['mentalRT-' num2str(iRun) '.csv'];
            grabString = 'taskM.RTTime:';
        case 'L'
            output = ['listRt-' num2str(iRun) '.csv'];
            grabString = 'taskL.RTTime:';
        case 'T'
            output = ['listRT-' num2str(iRun) '.csv'];
            grabString = 'taskT.RTTime:';
    end
end
txt = textscan(input, '%s', 'delimiter', '\n');

values = [];

for i=1:length(txt{1})
    txt{1}{i} = txt{1}{i}(find(~isspace(txt{1}{i})));
    if contains(txt{1}{i}, grabString)
        newStr = erase(txt{1}{i}, grabString);
        values = [values, str2num(newStr)];
    end
end

writematrix(values, output);
function output = generate_matrix(filename, taskName, onsetOrRT)
input = fopen(filename);
if onsetOrRT
    switch taskName
        case 'M'
            grabString = 'taskM.OnsetTime:';
        case 'L'
            grabString = 'taskL.OnsetTime:';
        case 'T'
            grabString = 'taskT.OnsetTime:';
    end
else
    switch taskName
        case 'M'
            grabString = 'taskM.RTTime:';
        case 'L'
            grabString = 'taskL.RTTime:';
        case 'T'
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

output = values;
end
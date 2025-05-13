% ------------------------------- %
%                                 
% Exercise 4: Hemodynamic Response and General Linear Model (GLM) 
%                                 
% ------------------------------- %
% ECE 8395-02, Analysis of fMRI Data - Fall 2023
% Vanderbilt University
% catie.chang@vanderbilt.edu
%
%
% Please download the following files:
% (1) fmri_blockDes.nii.gz (fMRI data that's already preprocessed with slice-timing
% correction and motion coregistration)
% (2) task-checkerboard_events.tsv
% While this is provided as an m-file, you are welcome to use Python.


%% Block-design task analysis
% ---------------------------- %
% Regressor construction
% ---------------------------- %
% The file "task-checkerboard_events.tsv" describes the stimuli
% presented during the fMRI scan corresponding to the data in 
% "fmri_blockDes.nii.gz".
%
% This was a block-design experiment, in which flashing
% checkerboards were presented in blocks of 20-second duration,
% with onset times (relative to the start of the scan) as specified in the
% "onset" column of this file. 
%
% (1) Based on the information in task-checkerboard_events.tsv,
% construct a regressor that we can use to investigate which voxels in the
% brain were significantly 'active' in response to the checkerboard stimuli.
% This regressor represents our model of the ideal task response, which we
% will take to be a binary stimulus waveform convolved with the HRF.
%
% You will need to check out the fMRI dataset to find  the
%  number of time frames and the TR:

hdr = niftiinfo('fmri_blockDes.nii.gz');
nframes = hdr.ImageSize(4);
TR = hdr.PixelDimensions(4);

% Programs such as FSL and SPM each contain a default HRF model, but
% for now we will use the following model, which is based on
% a double-gamma fit in Glover et al. 1999, "Deconvolution of
% impulse response in event-related BOLD fMRI", Neuroimage 9(4):416-29.
%
tmax = 30; % seconds (hrf duration)
t = [0:TR:tmax]; % sampled at TR
n1=5.0; t1=1.1; n2=12.0; t2 = 0.9; a2 = 0.4;
h1 = t.^n1.*exp(-t/t1);  
h2 = t.^n2.*exp(-t/t2);
h = h1/max(h1) - a2*h2/max(h2);
h = h/max(h);
% see what it looks like:
figure; plot(t,h); xlabel('time (s)'); title('HRF model');

% Now, build your regressor. The Matlab function "conv" will also be
% helpful. Make sure that the length of your regressor is equal 
% to the number of fMRI time frames.  
stimuli = readtable("task-checkerboard_events.tsv", 'FileType', 'text', 'Delimiter', '\t');
onset = stimuli.onset;
dur = stimuli.duration;
task = zeros(171,1);

for i=1:length(onset)
    if onset(i)/2 < length(task)
        task((onset(i)/2):(onset(i)/2)+10) = 1;
    end
end

predicted = conv(task, h);
predicted = predicted(1:171);


%% ---------------------------- %
% Design matrix
% ---------------------------- %
% (2) Form the design matrix, X, by appending a column of 1's to
% the regressor you made in step (1), along with linear and
% quadratic trends.
% For the interpretation of the betas
% (regression coefficients), it is often helpful to zero-mean the
% columns of X (except for the column of 1's). Why do you think this is
% helpful?  
X = [normalize(predicted), ones(171,1), normalize((1:171)'), normalize((1:171).^2)'];

% For fun, we can look at the design matrix
figure; imagesc(X); title('design matrix'); 

%% ---------------------------- %
% GLM
% ---------------------------- %
% (3) Find the matrix "Beta" that minimizes the squared error
% between the fMRI data and X*Beta. Just as in Exercise
% 2, it will help to rearrange the 4D fMRI data into a 2D (time x
% voxels) matrix.  If you do this, Beta will be a (4 x voxels)
% matrix.
brain_data = niftiread('fmri_blockDes.nii.gz');
Y = reshape(brain_data, [(64*64*32), 171])';
Y = double(Y);
B = X\Y;

% (4) Make a 3D "Beta Map", where the value at each voxel
% corresponds to its beta (parameter estimate) for the *stimulus
% regressor*. Then, display axial slices 10 and 19, choosing an
% informative color scale. [-20,20] may work well.
estimate = B(1, :);
estimate = reshape(estimate', [64,64,32]);

slice10 = estimate(:,:,10);
slice19 = estimate(:,:,19);

figure
subplot(1,2,1)
imshow(slice10, [])
title('Slice 10')
colormap(jet)
caxis([-20 20])
set(gcf, 'Position', [600, 600, 800, 600])

subplot(1,2,2)
imshow(slice19, [])
title('Slice 19')
colormap(jet)
caxis([-20 20])
saveas(gcf, 'beta_slices.png')


% (5) Plot the time course of a voxel in Slice 10 that has a high,
% positive beta value. You should be able to clearly see its
% response to the stimulus.
max_beta = max(slice10(:));
% max_response_index = find(estimate(:,:,10) == max_beta);
[x,y] = ind2sub( size(estimate(:,:,10)), find(estimate(:,:,10) == max_beta));
timeseries = squeeze(brain_data(x,y,10,:));

figure
plot(timeseries)
saveas(gcf, 'max_response.png')
%% ---------------------------- %
% Hypothesis testing
% ---------------------------- %
% (6) In order to test for significant task activation in the GLM
% framework, what would be our null hypothesis? 

% (7) Based on your design matrix, what would be the appropriate
%  contrast vector 'c' for this test? Enter it here:
c = [1; 0; 0; 0];

Beta = B;
% To perform classical, univariate parametric inference, we can form the
% t-statistic for each voxel as follows:
err = Y - X*Beta;  % if Y is your (time x voxels) matrix of fMRI data
n = size(X,1);
p = size(X,2);
errorVar = (1/(n-p))*sum(err.^2,1); % (n-p) degrees of freedom
t_num = c'*Beta; % numerator of t-statistic
t_den = sqrt(errorVar*(c'*inv(X'*X)*c)); % denominator of
                                    % t-statistic. Assumes c is a
                                    % column vector.
t_stat = t_num./t_den; % (1 x voxels) vector of t-statistics
t_vol = reshape(t_stat,hdr.ImageSize(1:3)); % 3D map of t-stats

% (8) For a 2-sided t-test, and with a t-distribution with (n-p)
% degrees of freedom, what is the critical t-value for alpha=0.001?
% A helpful matlab function is: tinv
df = n-p;
a = 0.001;
crit_t = tinv(1-a/2, df); % 3.3497 - divided by 2 since it?s a two-tailed test

% (9) How many voxels survive this threshold (positive and negative)? 
pos = sum(t_vol(:) >= crit_t); % 2694
neg = sum(t_vol(:) <= (-1*crit_t)); % 1244

% Note: this GLM analysis assumes that fMRI noise (residuals) are
% uncorrelated in time. However, some temporal autocorrelation is
% typically present, which would reduce the effective
% degrees-of-freedom at each voxel (and therefore impact the
% statistical testing). The best way to handle this is still a
% matter of discussion! 
% Here's a relevant blog post:
% https://mandymejia.wordpress.com/2016/11/06/how-to-efficiently-prewhiten-fmri-timeseries-the-right-way/




%% ---------------------------- %
% Initial time frames
% ---------------------------- %
% (10) Plot the time course of the randomly-selected voxel x,y,z =
% (20,20,12). As you can see (from this voxel, and the one you
% plotted earlier), the first several time frames have much much higher
% values than the rest of the signal. Why is this the case?  
random_timeseries = squeeze(brain_data(20,20,12,:));

figure
plot(random_timeseries)
saveas(gcf, "random_timeseries.png")

% To improve the fit of our GLM, we could have removed the first
% 4-5 time frames of the fMRI data and the first 4-5 time frames
% of our design matrix.







%% Function for

function beta_weights = step_2_perform_glm(brain_data, TR, nframes, task_regressor)
    
    % Programs such as FSL and SPM each contain a default HRF model, but
    % for now we will use the following model, which is based on
    % a double-gamma fit in Glover et al. 1999, "Deconvolution of
    % impulse response in event-related BOLD fMRI", Neuroimage 9(4):416-29.
    tmax = 30; % seconds (hrf duration)
    t = 0:TR:tmax; % sampled at TR
    n1=5.0; t1=1.1; n2=12.0; t2 = 0.9; a2 = 0.4;
    h1 = t.^n1.*exp(-t/t1);  
    h2 = t.^n2.*exp(-t/t2);
    h = h1/max(h1) - a2*h2/max(h2);
    h = h/max(h);

    predicted = conv(task_regressor, h);
    predicted = predicted(1:nframes);
    % figure; plot(predicted); ylim([-5,7]);
    
    X = [normalize(predicted), ones(nframes,1), normalize((1:nframes)'), normalize((1:nframes).^2)'];
    
    % For fun, we can look at the design matrix
    % figure; imagesc(X); title('design matrix'); 
    
    %% ---------------------------- %
    % GLM
    % ---------------------------- %
    % (3) Find the matrix "Beta" that minimizes the squared error
    % between the fMRI data and X*Beta. Just as in Exercise
    % 2, it will help to rearrange the 4D fMRI data into a 2D (time x
    % voxels) matrix.  If you do this, Beta will be a (4 x voxels)
    % matrix.
    
    Y = reshape(brain_data, [(91*109*91), nframes])';
    Y = double(Y);
    B = X\Y;
    
    estimate = B(1, :);
    beta_weights = reshape(estimate', [91,109,91]);

end







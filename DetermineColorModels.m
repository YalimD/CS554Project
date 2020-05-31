function [color_model] = DetermineColorModels(I, I2)

   % Get segmented ones
    [S1, M1] = segment(I);
    [S2, M2] = segment(I2);

    % Detect dense feature points
    imagePoints = detectMinEigenFeatures(rgb2gray(S1), 'MinQuality', 0.00001);

    % Create the point tracker
    tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

    % Initialize the point tracker
    imagePoints1 = imagePoints.Location;
    initialize(tracker, imagePoints1, S1);

    % Track the points
    [imagePoints2, validIdx] = step(tracker, S2);
    matchedPoints1 = imagePoints1(validIdx, :);
    matchedPoints2 = imagePoints2(validIdx, :);

    % show segments
    figure
    showMatchedFeatures(S1, S2, matchedPoints1, matchedPoints2);
    title('Segment Matches');

    figure;
    imshowpair(S1, S2, 'montage'); 
    title('Segmentations');

    %Normalize and turn pixels into features

    foreground = find(M1);
    background = find(~M1);

    I1_f = single(I) / 255;

    foreground_colors = [];
    background_colors = [];

    for c=1:3
        chan = I1_f(:,:,c);
        foreground_colors = [foreground_colors , chan(foreground)];
        background_colors = [background_colors , chan(background)];
    end

    %Define a gaussian for foreground
    foreground_model = fitgmdist(foreground_colors, 1);
    background_model = fitgmdist(background_colors, 1);
    
    means = [foreground_model.mu; background_model.mu];
    sigmas = zeros(3,3,2);
    sigmas(:,:,1) = foreground_model.Sigma;
    sigmas(:,:,2) = background_model.Sigma;
    
    color_model = gmdistribution(means, sigmas);

end

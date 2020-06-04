function [color_model] = DetermineColorModels(I, I2)

    %% Get segmented ones
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

    I1_RGB = single(I) / 255;
    I1_LAB = rgb2lab(I1_RGB);
    
    I1_LAB(:,:,1) = I1_LAB(:,:,1) / 100;
    I1_LAB(:,:,2) = (I1_LAB(:,:,2) + 100) / 200;
    I1_LAB(:,:,3) = (I1_LAB(:,:,3) + 100) / 200;

    foreground_colors = [];
    foreground_colors_rgb = [];
    
    background_colors = [];
    background_colors_rgb = [];

    for c=1:3
        chan = I1_LAB(:,:,c);
        foreground_colors = [foreground_colors , chan(foreground)];
        background_colors = [background_colors , chan(background)];
        
        chan_rgb = I1_RGB(:,:,c);
        foreground_colors_rgb = [foreground_colors_rgb , chan_rgb(foreground)];
        background_colors_rgb = [background_colors_rgb , chan_rgb(background)];
    end
    
    %% Visualize the colors
    %TODO: Visualize the model centers etc.
    figure;
    sizes = zeros(size(foreground_colors,1),1) + 10;
    colors = foreground_colors_rgb;
    scatter3(foreground_colors(:,1), foreground_colors(:,2), foreground_colors(:,3), sizes, colors);
    set(gca,'XLim',[0 1],'YLim',[-1 1],'ZLim',[-1 1]);
    
    xlabel("L");
    ylabel("A");
    zlabel("B");
    
    figure;
    sizes = zeros(size(background_colors,1),1) + 10;
    colors = background_colors_rgb;
    scatter3(background_colors(:,1), background_colors(:,2), background_colors(:,3), sizes, colors);
    set(gca,'XLim',[0 1],'YLim',[-1 1],'ZLim',[-1 1]);
    
    xlabel("L");
    ylabel("A");
    zlabel("B");
    
    %% Define a gaussian for foreground and background
    %TODO: Multiple components if necessary
    foreground_model = fitgmdist(foreground_colors, 1);
    background_model = fitgmdist(background_colors, 1);
    
    means = [foreground_model.mu; background_model.mu];
    sigmas = zeros(3,3,2);
    sigmas(:,:,1) = foreground_model.Sigma;
    sigmas(:,:,2) = background_model.Sigma;
    
    color_model = gmdistribution(means, sigmas);

end

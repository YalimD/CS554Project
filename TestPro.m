clear all;
close all;
clc;

useVideo = false;

if useVideo == false
    % Load camera parameters
    load('cameraParams.mat');

%     Load images from folder

    data_dir = 'test';
    pattern = dir(fullfile(data_dir, 'ab*.jpg'));

%     data_dir = 'CV_Data\\mov_cam';
%     pattern = dir(fullfile(data_dir, 'IMG*.jpg'));

    number_of_images = numel(pattern);
    images = cell(1, number_of_images);

    for im = 1:numel(pattern)

        file = fullfile(data_dir, pattern(im).name);
        images{im} = imread(file);
    end
else
    loaded = load('cameraParamsVideo.mat');
    cameraParams = loaded.cameraParamsVideo;

    vd = VideoReader('CV_Data\\video3.mp4');
    images = cell(1, 1000);

    number_of_images = 0;
    frame_step = 5;
    current_step = 0;

    while hasFrame(vd)
        if mod(current_step, frame_step) == 0
            image = readFrame(vd);
            number_of_images = number_of_images + 1;
            images{number_of_images} = image;

            current_step = 0;
        else
    %         Pass the frame
            readFrame(vd);
        end

        current_step = current_step + 1;

    end

end

fprintf('Number of images read %i \n', number_of_images);

progressed_images = 0;

%Process the first image to determine the initial features, segmentation and coordinates
I = images{1};
I = undistortImage(I, cameraParams, 'OutputView','valid');

% prevPoints = detectMinEigenFeatures(rgb2gray(I), 'MinQuality', 0.1);

% Create the point tracker
% tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);
% prevPoints = prevPoints.Location;
% initialize(tracker, prevPoints, I);

%Azdapted from SFM code
border = 50;
roi = [border, border, size(I, 2)- 2*border, size(I, 1)- 2*border];
prevPoints   = detectSURFFeatures(rgb2gray(I), 'NumOctaves', 8);

% Extract features. Using 'Upright' features improves matching, as long as
% the camera motion involves little or no in-plane rotation.
prevFeatures = extractFeatures(rgb2gray(I), prevPoints, 'Upright', true);

% Create a viewset that will maintain the camera configurations through
% images
viewSet = imageviewset;

viewId = 1;
%Rigid3d is a form that contains translation component like in Unity
viewSet = addView(viewSet, viewId, rigid3d, 'Points', prevPoints);

%Need at least 2 views for this, determine foreground and background models
%depending on the color distributions. Uses GMM
[color_model] = DetermineColorModels(images{1}, images{2});

figure;

%%For every pair of image (moving window):
% Find exterinsic camera configuration
for im = 2:number_of_images
    
    fprintf('Processing image %d \n', im);
    
    I = images{im};
    prevI = images{im-1};
    
    imshowpair(prevI, I, 'montage'); 
    
    % Undistort image using cameraParams, crop only valid pixels
    I = undistortImage(I, cameraParams,'OutputView','valid');

    currPoints   = detectSURFFeatures(rgb2gray(I), 'NumOctaves', 8);
    currFeatures = extractFeatures(rgb2gray(I), currPoints, 'Upright', true);    
    indexPairs   = matchFeatures(prevFeatures, currFeatures, ...
        'MaxRatio', .7, 'Unique',  true);
%     
%     % Track the points
%     [currPoints, validIdx] = step(tracker, I);
%     
%     % Select matched points.
%     matchedPointsPrev = prevPoints(validIdx,:);
%     matchedPointsCurr = currPoints(validIdx,:);

    % Select matched points.
    matchedPointsPrev = prevPoints(indexPairs(:, 1));
    matchedPointsCurr = currPoints(indexPairs(:, 2));

    %TODO: This is basically esential matrix version of our fundamental
    %matrix solution, most probably should be replaced
    [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
                                matchedPointsPrev, matchedPointsCurr, cameraParams);
    
    % Get the table containing the previous camera pose.
    prevPose = poses(viewSet, im-1).AbsolutePose;
    relPose  = rigid3d(relativeOrient, relativeLoc);
        
    % Compute the current camera pose in the global coordinate system 
    % relative to the first view.
    currPose = rigid3d(relPose.T * prevPose.T);
    
    % Add the current view to the view set.
    viewSet = addView(viewSet, im, currPose, 'Points', currPoints);

    % Store the point matches between the previous and the current views.
%     matches = repmat((1:size(prevPoints, 1))', [1, 2]);
%     matches = matches(validIdx, :);  
%     viewSet = addConnection(viewSet, im-1, im, relPose, 'Matches', matches);
    viewSet = addConnection(viewSet, im-1, im, relPose, 'Matches', indexPairs(inlierIdx,:));
    
    % Find point tracks across all views.
    tracks = findTracks(viewSet);

    % Get the table containing camera poses for all views.
    camPoses = poses(viewSet);
   
    % Triangulate initial locations for the 3-D world points.
    xyzPoints = triangulateMultiview(tracks, camPoses, cameraParams);
    
    % Refine the 3-D world points and camera poses.
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
        'PointsUndistorted', true);

    % Store the refined camera poses.
    viewSet = updateView(viewSet, camPoses);
    
    prevPoints  = currPoints;  
    prevFeatures = currFeatures;
    
    progressed_images = progressed_images + 1;

end  

% Display camera poses.
camPoses = poses(viewSet);
figure;
plotCamera(camPoses, 'Size', 0.2);
hold on

% Exclude noisy 3-D points.
goodIdx = (reprojectionErrors < 20);
xyzPoints = xyzPoints(goodIdx, :);

% Display the 3-D points.
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on
hold off

% Specify the viewing volume.
% TODO: GOTTA SPECIFY THIS BETTER
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-10, loc1(1)+10]);
ylim([loc1(2)-10, loc1(2)+10]);
zlim([loc1(3)-10, loc1(3)+30]);
camorbit(0, -30);

title('Refined Camera Poses');

%% Volume fitting

images = images(1:progressed_images+1);

%%Gonna get multiple camera configurations for every image after first ones
%TODO: IT MIGHT ALSO BE BETTER TO SEND THE VOLUME LOCATIONS RELATIVE TO
%CAMERA POSITIONS
Volume(images, color_model, cameraParams, viewSet);
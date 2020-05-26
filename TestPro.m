clear all;
close all;
clc;

%%TODO: Do this for video where multiple frames exist

% Load camera parameters
load('cameraParams.mat');

% Load images from folder
data_dir = 'CV_Data\\mov_cam';
pattern = dir(fullfile(data_dir, 'IMG*.jpg'));

data_dir = 'test';
pattern = dir(fullfile(data_dir, 'ab*.jpg'));

number_of_images = numel(pattern);
images = cell(1, number_of_images);

for im = 1:numel(pattern)
    
    file = fullfile(data_dir, pattern(im).name);
    images{im} = imread(file);
end


loaded = load('cameraParamsVideo.mat');
cameraParams = loaded.cameraParamsVideo;

vd = VideoReader('CV_Data\\video3.mp4');
images = cell(1, 1000);

number_of_images = 0;
frame_step = 30;
current_step = 0;

while hasFrame(vd)
    if mod(current_step, frame_step) == 0
        image = readFrame(vd);
        number_of_images = number_of_images + 1;
        images{number_of_images} = image;
        
        current_step = 0;
    else
        %Pass the frame
        readFrame(vd);
    end
        
    current_step = current_step + 1;
   
end

fprintf('Number of images read %i', number_of_images);

%%For every pair of image (moving window):
% Find foreground image and mask
% Find exterinsic camera configuration

latest_camera_R = eye(3);
latest_camera_t = [0,0,0];

camera_rotations = cell(1, number_of_images);
camera_translations = cell(1, number_of_images);

camera_rotations{1} = latest_camera_R;
camera_translations{1} = latest_camera_t;

progressed_images = 0;

for im = 1:number_of_images-1
    
    I1 = images{im};
    I2 = images{im + 1};
    
    % Display original images
%     figure;
%     imshowpair(I1, I2, 'montage'); 
%     title('Original Images');

    % Undistort images using cameraParams, crop only valid pixels
    I1 = undistortImage(I1, cameraParams,'OutputView','valid');
    I2 = undistortImage(I2, cameraParams,'OutputView','valid');

%     % Display undistorted images
%     figure;
%     imshowpair(I1, I2, 'montage');
%     title('Undistorted Images');

    % Detect feature points on I1
    imagePoints1 = detectMinEigenFeatures(rgb2gray(I1), 'MinQuality', 0.1);

    % Visualize detected points
    figure;
    imshow(I1);
    title('Feature Points in First Image');
    hold on;
    plot(imagePoints1);

    % Create the point tracker
    tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

    % Initialize the point tracker
    imagePoints1 = imagePoints1.Location;
    initialize(tracker, imagePoints1, I1);

    % Track the points
    [imagePoints2, validIdx] = step(tracker, I2);
    matchedPoints1 = imagePoints1(validIdx, :);
    matchedPoints2 = imagePoints2(validIdx, :);

    % Visualize correspondences
%     figure
%     showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
%     title('Tracked Features');
    
    if size(matchedPoints1,1) < 8
        break;
    end

    % Estimate the fundamental matrix
    [fMatrix, epipolarInliers] = estimateFundamentalMatrix(...
      matchedPoints1, matchedPoints2, 'Method', 'MSAC', 'NumTrials', 100000);

    % Find epipolar inliers
    inlierPoints1 = matchedPoints1(epipolarInliers, :);
    inlierPoints2 = matchedPoints2(epipolarInliers, :);

    % Display inlier matches
    figure
    showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
    title('Epipolar Inliers');

    [R, t] = relativeCameraPose(fMatrix, cameraParams, inlierPoints1, inlierPoints2);
    
    % Some poses give 4 different solutions, where we only use the first
    % one
    if size(R,3) > 1
        R = R(:,:,1);
        t = t(1,:);
    end
    
    %% Segmentation part, it should be enough that we do this par tonlya nd only once
    if im == 1

        % Get segmented ones
        [S1, M1] = segment(I1);
        [S2, M2] = segment(I2);

        % Detect dense feature points
        imagePoints1 = detectMinEigenFeatures(rgb2gray(S1), 'MinQuality', 0.00001);

        % Create the point tracker
        tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

        % Initialize the point tracker
        imagePoints1 = imagePoints1.Location;
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
        
        I1_f = single(I1) / 255;
        
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
    end

    % Compute the camera matrices for each position of the camera
    % The very first camera is at the origin looking along the X-axis. Thus, its
    % rotation matrix is identity, and its translation vector is 0.
    
    %For the rest of the cameras, latest camera R and t are recorded
    
    %Load the camera matrix for the first image
    latest_camera_R = camera_rotations{im};
    latest_camera_t = camera_translations{im};
    
    camMatrix1 = cameraMatrix(cameraParams, latest_camera_R, latest_camera_t);
    
    latest_camera_R = R * latest_camera_R;
    latest_camera_t = t * latest_camera_R + latest_camera_t;
    
    camMatrix2 = cameraMatrix(cameraParams, latest_camera_R, latest_camera_t);
  
    camera_rotations{im + 1} = latest_camera_R;
    camera_translations{im + 1} = latest_camera_t;
    
    progressed_images = progressed_images + 1;
    
    close all;
    
end  
%% FOR DEBUGGING ONLY, THESE POINTS WON'T BE USED
% Compute the 3-D points
points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

% Get the color of each reconstructed point
numPixels = size(S1, 1) * size(S1, 2);
allColors = reshape(S1, [numPixels, 3]);
colorIdx = sub2ind([size(S1, 1), size(S1, 2)], round(matchedPoints1(:,2)), ...
    round(matchedPoints1(:, 1)));
color = allColors(colorIdx, :);

% Create the point cloud
% points3D = [0,0,0];
% color = [0,0,0];

ptCloud = pointCloud(points3D, 'Color', color);

camera_rotations = camera_rotations(1:progressed_images+1);
camera_translations = camera_translations(1:progressed_images+1);


figure
grid on
cameraSize = 0.3;
for cam = 1:progressed_images+1
    % Visualize the camera locations and orientations

    if cam == im
        c_color = 'r';
    else
        c_color = 'b';
    end

    cam_R = camera_rotations{cam};
    cam_t = camera_translations{cam};

    plotCamera('Location', cam_t, 'Orientation', cam_R,...
        'Size', cameraSize, 'Color', c_color, 'Label', num2str(cam), 'Opacity', 0);
    hold on

end

    % Visualize the point cloud
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);

% Rotate and zoom the plot
camorbit(0, -30);
camtarget([25, 0, 25]); % camzoom(1.5);

% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')

title('Up to Scale Reconstruction of the Scene');


%% Volume fitting

images = images(1:progressed_images+1);
masks = {foreground_model, background_model};
camera_extrinsics = {camera_rotations, camera_rotations};

%%Gonna get multiple camera configurations for every image after first ones
Volume(images, masks, cameraParams, camera_extrinsics);
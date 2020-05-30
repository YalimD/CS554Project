function [log_prob] = CalculatePixelProb(voxel_count, bounds, images, color_model,...
    camera_intrinsic, cameras)

    x_bounds = bounds{1};
    y_bounds = bounds{2};
    z_bounds = bounds{3};

    log_prob = zeros(voxel_count,voxel_count,voxel_count);    
    world_points = [x_bounds(:), y_bounds(:), z_bounds(:)];
    
    obj_probs = ones(size(world_points,1),1);
    back_probs = ones(size(world_points,1),1);
   
    camPoses = poses(cameras);
    
    number_of_cams = size(camPoses,1);
    
    %For each camera, project and btain the pixel
    %TODO: Size is differenet
    for cam_i=1:number_of_cams
        
        rot = camPoses.AbsolutePose(cam_i).Rotation;
        trans = camPoses.AbsolutePose(cam_i).Translation;
        
        % DONT FORGET, THE AXIS OF 3D SPACE IS WHERE Y IS UP.
        image_points = worldToImage(camera_intrinsic, rot, trans, world_points);
        
        current_image = images{cam_i};
        
        % Filter out of image points
        lowerLimit = and(image_points(:,1) >= 1,image_points(:,2) >= 1);
        upperLimit = and(image_points(:,1) <= size(current_image,1), ...
                         image_points(:,2) <= size(current_image,2));
        withinImageFilter = and(lowerLimit, upperLimit);
        
        image_points = round(image_points);
        image_points = int32(image_points);
        filtered_image_points = image_points(withinImageFilter,:);
        
        %Convert the positions into indices
        image_points_red = sub2ind(size(current_image), filtered_image_points(:,1), ...
                                                     filtered_image_points(:,2), ones(size(filtered_image_points,1),1));
        image_points_green = sub2ind(size(current_image), filtered_image_points(:,1), ...
                                                     filtered_image_points(:,2), ones(size(filtered_image_points,1),1) + 1);
        image_points_blue = sub2ind(size(current_image), filtered_image_points(:,1), ...
                                                     filtered_image_points(:,2), ones(size(filtered_image_points,1),1) + 2);
         
        pixels = zeros(size(filtered_image_points,1), 3);
        pixels(:,1) = current_image(image_points_red);
        pixels(:,2) = current_image(image_points_green);
        pixels(:,3) = current_image(image_points_blue);
        
%         temp_obj_probs = ones(obj_probs);
%         temp_obj_probs(withinImageFilter) = posterior(foreground_model, pixels);
%         temp_back_probs = ones(back_probs);
%         
%         obj_probs = temp_obj_probs;
        
        pixels = pixels / 255;
        
        probs = posterior(color_model, pixels);
        foreground_probs = probs(:,1);
        background_probs = probs(:,2);

        obj_probs(withinImageFilter) = obj_probs(withinImageFilter) .* foreground_probs;
        back_probs(withinImageFilter) = back_probs(withinImageFilter) .* (1 - background_probs);

    end

    %Eps to prevent 0 cases
    obj_probs = nthroot(obj_probs, number_of_cams) + eps;
    back_probs = 1 - (nthroot(back_probs, number_of_cams)) + eps;
               
    log_prob_2d = log(back_probs ./ obj_probs);
    
    i = 1;
    %TODO: This log prob form must be converted to 3d voxel model
    for v_z=1:voxel_count
        for v_x=1:voxel_count
            for v_y=1:voxel_count
                log_prob(v_x,v_y,v_z) = log_prob_2d(i);
                i = i + 1;
            end
        end
    end

end
function [log_prob] = CalculatePixelProb(voxel_count, bounds, images, masks,...
    camera_intrinsic, cameras)

    x_bounds = bounds{1};
    y_bounds = bounds{2};
    z_bounds = bounds{3};

    log_prob = zeros(voxel_count,voxel_count,voxel_count);
       
    foreground_model = masks(1);
    background_model = masks(2);
    
    world_points = [x_bounds(:), y_bounds(:), z_bounds(:)];
    
    obj_probs = ones(voxel_count,voxel_count,voxel_count);
    back_probs = ones(voxel_count,voxel_count,voxel_count);
    
    number_of_cameras = size(cameras,2);
    
    %For each camera, project and btain the pixel
    %TODO: Size is differenet
    for cam_i=1:size(cameras)
        
        rot = cameras.AbsolutePose(cam_i).Rotation;
        trans = cameras.AbsolutePose(cam_i).Translation;
        
        % DONT FORGET, THE AXIS OF 3D SPACE IS WHERE Y IS UP.
        image_points = worldToImage(camera_intrinsic, rot, trans, world_points);
        
        image_points = sub2ind(size(image_points,1), image_points(:,1), ...
                                                     image_points(:,2));

        image = images(cam_i);
        
        pixels = image(image_points);
        
        obj_probs = obj_probs * ...
            posterior(foreground_model, pixels);
        
        back_probs = back_probs * ...
             (1 - mvnpdf(background_model, pixels));

    end
    
    obj_probs = sqrt(obj_probs) * number_of_cameras;
    back_probs = 1 - sqrt(obj_probs) * number_of_cameras;
                
    log_probs = log(back_probs / obj_probs);

end
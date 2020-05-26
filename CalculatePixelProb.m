function [log_prob] = CalculatePixelProb(voxel_count, bounds, images, masks,...
    camera_intrinsic, camera_extrinsics)

    x_bounds = bounds(1);
    y_bounds = bounds(2);
    z_bounds = bounds(3);

    log_prob = zeros(voxel_count,voxel_count,voxel_count);
    
    cam_rots = camera_extrinsics(1);
    cam_trans = camera_extrinsics(2);
    
    foreground_model = masks(1);
    background_model = masks(2);
    
    world_points = [x_bounds(:), y_bounds(:), z_bounds(:)];
    
    obj_probs = ones(voxel_count,voxel_count,voxel_count);
    back_probs = ones(voxel_count,voxel_count,voxel_count);
    
    number_of_cameras = size(camera_extrinsics,2);
    
    %For each camera, project and btain the pixel
    for cam_i=1:size(camera_extrinsics,2)
        
        rot = cam_rots(cam_i);
        trans = cam_trans(cam_i);

        image_points = worldToImage(camera_intrinsic, rot, trans, world_points);
        
        image_points = sub2ind(size(image_points,1), image_points(:,1), ...
                                                     image_points(:,2));

        image = images(cam_i);
        
        pixels = image(image_points);
        
        
        %% TODO: These should be fixed to 1
        obj_probs = obj_probs * ...
            mvnpdf(pixels, foreground_model.mu, foreground_model.Sigma);
        
        back_probs = back_probs * ...
             (1 - mvnpdf(pixels, background_model.mu, background_model.Sigma));

    end
    
    obj_probs = sqrt(obj_probs) * number_of_cameras;
    back_probs = 1 - sqrt(obj_probs) * number_of_cameras;
                
    log_probs = log(back_probs / obj_probs);

end

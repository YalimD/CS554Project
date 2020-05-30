function [u_volume] = Volume(images, color_model, camera_intrinsics, cameras)
    
    %Create the volume
    %Initialize the surface variables (u) for each voxel to 0.5
    %that means equal probability of being either background or foregorund
    %%TODO: We might also increase this towards center of the volume
    
    %TODO: Voxel must be placed such that its center is 0,0,0. This is
    %necessary for it to be compliant with the camera extrinsic parameters
    voxel_count = 100;
    u_volume = zeros(voxel_count,voxel_count,voxel_count) + 0.5;
    u_volume_next = u_volume;
    
    u_bar_volume = u_volume;
    
    %In the camera configurations, first camera is at 000 looking over X
    %axis. The system is right handed, Z being inwards
    x_bounds = [-10,10];
    x_step = (x_bounds(2) - x_bounds(1)) / voxel_count;
    
    y_bounds = [-10,10];
    y_step = (y_bounds(2) - y_bounds(1)) / voxel_count;
    
    z_bounds = [10,30];
    z_step = (z_bounds(2) - z_bounds(1)) / voxel_count;
    
    bounds = {x_bounds, y_bounds, z_bounds};
    
    [pos_volumes_x pos_volumes_y pos_volumes_z]  = meshgrid(x_bounds(1):x_step: x_bounds(2) - x_step, ...
                                        y_bounds(1):y_step: y_bounds(2) - y_step, ...
                                        z_bounds(1):z_step: z_bounds(2) - z_step);
                                    
                                    
                    
    
    % Put gaussian noise into xi vector field components
    %TODO: Try other configuraions
    xi_x = rand(voxel_count,voxel_count,voxel_count);
    xi_y = rand(voxel_count,voxel_count,voxel_count);
    xi_z = rand(voxel_count,voxel_count,voxel_count);
    
    next_xi_x = xi_x;
    next_xi_y = xi_y;
    next_xi_z = xi_z;
    
    % Recommended in the paper, subject the change
    sigma = 0.1;
    tau = 0.1;
    nu = 1.8;
    
    %%Iterative process for energy minimization
    for n=1:1:1000
        
        %For each voxel in the volume, calculate the divergence and
        %graident according to u and xi
        
        %% Update the xi component
        [mag, az, elev] = imgradient3(u_bar_volume);
        
        %Using the azimuth and elevation, calculate gradient vectors
        [grad_x, grad_y, grad_z] = sph2cart(az, elev, mag);
        
        grad_x = grad_x * sigma;
        grad_y = grad_y * sigma;
        grad_z = grad_z * sigma;
        
        %Add to the previous xi
        xi_x = xi_x + grad_x;
        xi_y = xi_y + grad_y;
        xi_z = xi_z + grad_z;
        
        %Normalize
        for r=1:voxel_count
            for c=1:voxel_count
                for d=1:voxel_count
                    
                    xi = norm([xi_x(r,c,d), xi_y(r,c,d), xi_z(r,c,d)]);
                    
                    xi_x(r,c,d) = xi_x(r,c,d) / xi;
                    xi_y(r,c,d) = xi_y(r,c,d) / xi;
                    xi_z(r,c,d) = xi_z(r,c,d) / xi;
                end
            end
        end
        
        %% Update u
        div = divergence(pos_volumes_x, pos_volumes_y, pos_volumes_z, ...
            xi_x, xi_y, xi_z);
               
        %Find probs that pixel belongs to foregorund of background
        log_prob = CalculatePixelProb(voxel_count, {pos_volumes_x pos_volumes_y pos_volumes_z}, ...
            images, color_model, camera_intrinsics, cameras);
        
        update_term = tau * (nu * div - log_prob);
        
        u_volume_next = u_volume + update_term;
        
        u_volume_next(u_volume_next > 1) = 1;
        u_volume_next(u_volume_next < 0) = 0;
        
        u_bar_volume = 2 * u_volume_next - u_volume;
        
        u_volume = u_volume_next;
        
        %TODO: Render the latest volume
        
        fprintf('Done with iteration %d \n', n);
        
    end


end




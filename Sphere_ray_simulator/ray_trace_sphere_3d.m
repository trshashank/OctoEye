function [Retina_image,Retina_ang_ind,Retina_plane_vec] = ray_trace_sphere_3d(Source_location,Lens_radius,n_vals,Retina_min,Retina_max,pupil_file_path,do_3d_plot,do_intensity_plot,N_rays,line_color)

x0 = Source_location(2); %[cm] - x location of the point source
y0 = Source_location(3); %[cm] - y location of the point source
z0 = Source_location(1); %[cm] - location of the point source on the optical axis z
R = Lens_radius;         %[cm] - Radius of the spherical lens
RR = Retina_max;         %[cm] - Maximal distance of retina surface from sphere lens centre
RR_min = Retina_min;     %[cm] - Minimal distance of retina surface
Retina_ang_lim = pi/4; % Angular distribution of the retina surface (from -Retina_ang_lim to +Retina_ang_lim)

n1 = n_vals(3);         % Refractive index on the surface of the sphere
n_core = n_vals(1);     % Sphere core refractive index
n_external = n_vals(2); % Refractive index of the medium

r_vec = linspace(0,R,200);
n2_func = n_core*(1+(n_core/n_external - 1)*(r_vec/R).^2).^-1; % profile of refractive index
n_temp_factor = R^2 * n_external / (n_core - n_external);      % temporary factor used in computation of the local refractive index gradient
n2 = n_external;
dl = R/200;     % graded index simulation step;

% simulation resolution values:
N_rays_dir = round(sqrt(N_rays));   % number of rays in each axis (total number of rays is N^2, but many don't cross the pupil)
N_pup_dir = 201;                    % The resolution of the pupil image
N_retina_angs = 751;                % The resolution of the retina output image (angles!)
N_focal_surfaces = 100;             % Number of focal planes to calculate for

Retina_plane_vec = linspace(RR_min,RR,N_focal_surfaces);
Retina_image = zeros(N_retina_angs,N_retina_angs,N_focal_surfaces);
Retina_ang_ind = linspace(-Retina_ang_lim,Retina_ang_lim,N_retina_angs);

Intensity = N_rays_dir^-2;          % the intensity of a single ray
% smoothing kernel for retina image
ker = exp(-[-4:4].^2/4)'*exp(-[-4:4].^2/4);
ker = ker/sum(ker(:));


% take pupil image and turn it into binary transparent image + image for
% plotting the 3D sphere
pupil_from_file = rot90(rot90(imread(pupil_file_path))); % Load a binary image (0s and 1s)
pupil_binary = false(N_pup_dir,N_pup_dir);
pupil_image = false(size(pupil_from_file,1),size(pupil_from_file,2));
pupil_image(pupil_from_file(:,:,1)>20) = true;
pupil_image = double(imresize(pupil_image, size(pupil_binary),"nearest")); % Resize for mapping
pupil_binary = pupil_image>0;
pupil_image(pupil_image>0) = nan;


% do basic ploting
if do_3d_plot
    R_steps = linspace(R, 0, 30); % 10 shells from R to 0
    figure(2); hold on;
    for i = 1:length(R_steps)
        [X, Y, Z] = sphere(50); % Lower resolution for performance
        X = R_steps(i) * X; Y = R_steps(i) * Y; Z = R_steps(i) * Z;
        h = surf(X, Y, Z); 
        alpha_val = 0.05 + 0.1 * (i / length(R_steps)); % Gradually increase opacity
        set(h, 'FaceColor', [0.3 0.5 1], 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
    end

    [theta, phi] = meshgrid(linspace(-pi/2, pi/2, size(pupil_image,1)), linspace(3*pi/2, pi/2, size(pupil_image,2)));
    Y_hemi = R * cos(theta) .* sin(phi);
    X_hemi = R * sin(theta);
    Z_hemi = R * cos(theta) .* cos(phi);

    % Plot the sphere and map the pupil image
    pupil_image(pupil_image>0) = nan;
    figure(2);
    h = surf(Z_hemi, X_hemi, Y_hemi, pupil_image,'EdgeColor', 'none','CData', pupil_image); colormap gray
    grid on;

    [theta, phi] = meshgrid(linspace(-pi/2, pi/2, 50), linspace(3*pi/2, pi/2, 50));
    Y_hemi = Retina_plane_vec(end) * cos(theta) .* sin(phi);
    X_hemi = Retina_plane_vec(end) * sin(theta);
    Z_hemi = - Retina_plane_vec(end) * cos(theta) .* cos(phi);

    % Plot the sphere and map the pupil image
    % pupil_image(pupil_image>0) = nan;
    figure(2);
    h = surf(Z_hemi, X_hemi, Y_hemi, 'LineStyle',':','EdgeAlpha',0.2); colormap gray
    grid on;

    Axis_pointing_length = 1.6*R;
    plot([0,Axis_pointing_length,0],[0,0,0],'k'); hold on; text(0,Axis_pointing_length,0,'X'); 
    plot([0,0,Axis_pointing_length],[0,0,0],'k'); text(0,0,Axis_pointing_length,'Y');
    plot([Axis_pointing_length,0,0],[0,0,0],'k'); text(1.3*Axis_pointing_length,1.5,0,'Z');

    xlim([-5*R,RR+1])
    ylim([-1.5*R, 1.5*R])
    zlim([-1.5*R, 1.5*R])
end

% start ray simulation
min_ang_ph = atan((x0+R)/z0);
max_ang_ph = atan((x0-R)/z0);
ph = linspace(min_ang_ph,max_ang_ph,N_rays_dir);
dph = ph(2)-ph(1);
min_ang_te = atan((y0+R)/z0);
max_ang_te = atan((y0-R)/z0);
te = linspace(min_ang_te,max_ang_te,N_rays_dir);
dte = te(2)-te(1);

%% full simulation

X_0 = [z0,x0,y0];

% loop on all angles of incoming rays
for ph_i = 1:length(ph)
    for te_i = 1:length(te)
        ez = cos(ph(ph_i))*cos(te(te_i));
        ex = sin(ph(ph_i));
        ey = cos(ph(ph_i))*sin(te(te_i));
        e_0 = [ez,ex,ey];

        % Interface 1 - locate the ray intersecting the first sphere surface
        % eqns = [X_0(1)+a*e_0(1) == R*cos(phi2),X_0(2)+a*e_0(2) == R*sin(phi2),phi2>=pi - pup_ang,phi2<=pi + pup_ang];
        poliABC = [1, 2*(ex*x0 + ey*y0 + ez*z0), x0^2 + y0^2 + z0^2 - R^2];
        a_an = min(roots(poliABC));


        if isreal(a_an) % Does the ray hit the sphere?
            % X_1 -> point on sphere in first ray intersection
            X_1 = X_0 + a_an*e_0; % interface 1 location


            en_1 = X_1 / norm(X_1); % norm of incoming beam interface 1
            temp_theta = asin(X_1(2)/R);
            ind_x = round((temp_theta/pi+0.5)*(N_pup_dir-1)+1);
            temp_phi = asin(X_1(3)/R/cos(temp_theta));
            ind_y = round((temp_phi/pi+0.5)*(N_pup_dir-1)+1);

            if pupil_binary(ind_y,ind_x) % Does the ray cross the pupil?
                if do_3d_plot % plot incoming ray
                    figure(2)
                    line([X_0(1),X_1(1)],[X_0(2),X_1(2)],[X_0(3),X_1(3)],'Color',line_color)
                end

                % Senll's law on first interface
                beta_outside = acos(-dot(en_1,e_0));
                beta_inside = asin(n1/n2*sin(beta_outside)); % refraction angle
                e_1 = n1/n2 * e_0 + (n1/n2 * cos(beta_outside) - cos(beta_inside)) * en_1; % - for n1/n2=1 we must get e_1 = e_0 <<<<<<<<<<<<<<<<<<<<<<
                e_1 = e_1/norm(e_1);

                % start working on graded index ray bending
                e_in = e_1;
                X_a = X_1;
                ra = sum(X_a.^2)^0.5;
                na = n2_func(end);

                X_b = X_a + dl*e_1;
                rb = norm(X_b);
                [~,II] = min((R-rb)^2); % find index of radius vector nearest to ray location
                nb = n2_func(II);       % value of refractive index in new ray location
                if do_3d_plot
                    X_curves = nan(3,20);
                    r_draw = -R;
                    ind_draw = 1;
                end

                while rb<R % calculating new ray positions bending through the sphere, until reaching the other side.
                    if do_3d_plot
                        if X_b(1)>r_draw
                            X_curves(:,ind_draw) = X_a;
                            ind_draw = ind_draw + 1;
                            r_draw = r_draw+R/10;
                        end
                    end

                    mid_point = (X_a + X_b)/2; % refractive index in the current location of the ray
                    grad_n = - mid_point * (n_temp_factor * n_core) / (n_temp_factor + (ra + rb)/2)^2; % the refractive index gradient
                    e_in = e_in + dl * (2/(nb+na)) * grad_n;
                    e_in = e_in / norm(e_in); % corrected ray direction

                    X_a = X_b;
                    X_b = X_b + dl * e_in; % next location of the ray in the sphere
 
                    ra = rb;
                    rb = norm(X_b);
                    [~,II] = min((R-rb)^2);
                    na = nb;
                    nb = n2_func(II);

                end

                if do_3d_plot % plot the ray in the sphere
                    plot3([X_curves(1,~isnan(X_curves(1,:))),X_b(1)],[X_curves(2,~isnan(X_curves(2,:))),X_b(2)],[X_curves(3,~isnan(X_curves(3,:))),X_b(3)],'Color',line_color);
                end

                % X_2 -> point on sphere in second ray intersection
                X_2 = X_b;
                en_2 = X_2/norm(X_2);


                % Senll's law on second interface
                beta_inside = acos(dot(e_in,en_2));
                beta_outside = asin(n1/n2*sin(beta_outside)); % refraction angle
                if beta_outside < 1
                    e_2 = n2/n1 * e_in + (n2/n1 * cos(beta_inside) - cos(beta_outside)) * en_2;
                    e_2 = e_2/norm(e_2);

                    % ray incident on retina
                    for gi = 1:length(Retina_plane_vec)
                       
                        poliABC = [1, 2*sum(e_2.*X_2), R^2 - Retina_plane_vec(gi)^2];
                        solution_r = roots(poliABC);
                        b_inbetween = max(solution_r);
                        
                        % X_3 -> point on retina of the ray
                        X_3 = X_2 + e_2 * b_inbetween;

                        ang_x = atan(X_3(2)/X_3(1));
                        ang_y = atan(X_3(3)/X_3(1));
                        if abs(ang_x)<Retina_ang_lim && abs(ang_y)<Retina_ang_lim
                            [~,Ix] = min((ang_x-Retina_ang_ind).^2);
                            [~,Iy] = min((ang_y-Retina_ang_ind).^2);
                            Retina_image(Ix,Iy,gi) = Retina_image(Ix,Iy,gi) + Intensity;
                        end
                    end

                    if do_3d_plot % plot ray towards retina plane
                        figure(2)
                        line([X_2(1),X_3(1)],[X_2(2),X_3(2)],[X_2(3),X_3(3)],'Color',line_color)

                    end
                end

                if do_3d_plot % just alignment of the 3d plot and plot it while simulating 
                    figure(2)
                    xlim([-5*R,RR+1])
                    ylim([-1.5*R, 1.5*R])
                    zlim([-1.5*R, 1.5*R])

                    axis equal
                    drawnow;
                end
            end
        end
    end
end

% smooth the retina image (to reduce stipe artifacts caused
% when not enough rays are used, or rounding edges align) 
for si = 1:N_focal_surfaces
    Retina_image(:,:,si) = conv2(Retina_image(:,:,si),ker,'same');
end

% plot the retina intensity for various retina distances
if do_intensity_plot
    figure(44);
    colormap gray
    for si = 1:N_focal_surfaces
        imagesc(180*Retina_ang_ind/pi,180*Retina_ang_ind/pi,Retina_image(:,:,si)');
        title(['Energy on retina at ',num2str(Retina_plane_vec(si)),'cm'])
        xlabel('Horizontal X angle [deg]')
        ylabel('Vertical Y angle [deg]')
        colorbar
        drawnow;
        if si==1
            pause(2)
        else
            pause(0.1)
        end
    end
end
%%

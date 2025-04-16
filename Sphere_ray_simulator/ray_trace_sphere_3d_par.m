function [Retina_image_out,Retina_ang_ind,Retina_plane_vec] = ray_trace_sphere_3d_par(Source_location,Lens_radius,n_vals,Retina_min,Retina_max,pupil_file_path,N_rays)

x0 = Source_location(2);
y0 = Source_location(3);
z0 = Source_location(1); %cm
% x0 = 2; %cm
% y0 = 5;
% z0 = -80; %cm
R = Lens_radius;%3; %cm
RR = Retina_max;%R:0.5:50;
RR_min = Retina_min;
Retina_ang_lim = pi/4; % Angular distribution of the retina surface (from -Retina_ang_lim to +Retina_ang_lim)

n1 = n_vals(3);
n_core = n_vals(1);
n_external = n_vals(2);
temp1 = R^2 * n_external / (n_core - n_external);

% n_core = 1.27;%1.34;
% etta = 0.091;
r_vec = linspace(0,R,200);
n2_func = n_core*(1+(n_core/n_external - 1)*(r_vec/R).^2).^-1;
n2 = n_external;%n2_func(end);%1.34;
dl = R/200; %graded index simulation step;


N_rays_dir = round(sqrt(N_rays));%750;       % number of rays in each axis (total number of rays is N^2, but many don't cross the pupil)
N_rays = N_rays_dir^2;

N_pup_dir = 201;        % The resolution of the pupil image
N_retina_angs = 751;    % The resolution of the retina output image (angles!)
N_focal_surfaces = 100;  % Number of focal planes to calculate for

Intensity = N_rays_dir^-2;
% smoothing kernel for retina image
ker = exp(-[-4:4].^2/4)'*exp(-[-4:4].^2/4);
ker = ker/sum(ker(:));

Retina_plane_vec = linspace(RR_min,RR,N_focal_surfaces);
Retina_ang_ind = linspace(-Retina_ang_lim,Retina_ang_lim,N_retina_angs);

% take pupil image and turn it into binary transparent image + image for
% plotting the 3D sphere
pupil_from_file = rot90(rot90(imread(pupil_file_path))); % Load a binary image (0s and 1s)
pupil_binary = false(N_pup_dir,N_pup_dir);
pupil_image = false(size(pupil_from_file,1),size(pupil_from_file,2));
pupil_image(pupil_from_file(:,:,1)>20) = true;
pupil_image = double(imresize(pupil_image, size(pupil_binary),"nearest")); % Resize for mapping
pupil_binary = pupil_image>0;
pupil_image(pupil_image>0) = nan;

% start ray simulation
min_ang_ph = atan((x0+R)/z0);
max_ang_ph = atan((x0-R)/z0);
ph = linspace(min_ang_ph,max_ang_ph,N_rays_dir);
dph = ph(2)-ph(1);
min_ang_te = atan((y0+R)/z0);
max_ang_te = atan((y0-R)/z0);
te = linspace(min_ang_te,max_ang_te,N_rays_dir);
dte = te(2)-te(1);
%
% % int_fac = abs(sin(alpha+dalpha/2).^2 - sin(alpha-dalpha/2).^2);
% xg = Inf(length(RR),length(ph));
% image_x = linspace(-3/4*R,3/4*R,400);
% enrgy = zeros(length(RR),length(image_x));
% gi = length(RR);

%% full sim
% syms phi2 a
X_0 = [z0,x0,y0];
[PH,TE] = meshgrid(ph,te);
Dirs(:,1) = PH(:);
Dirs(:,2) = TE(:);
% gpuArray(Dirs);

Retina_image = cell(N_rays,1);

X_3 = [0,0,0];

parfor angle_ind = 1:N_rays

    ez = cos(Dirs(angle_ind,1))*cos(Dirs(angle_ind,2));
    ex = sin(Dirs(angle_ind,1));
    ey = cos(Dirs(angle_ind,1))*sin(Dirs(angle_ind,2));
    e_0 = [ez,ex,ey];

    % Interface 1
    % eqns = [X_0(1)+a*e_0(1) == R*cos(phi2),X_0(2)+a*e_0(2) == R*sin(phi2),phi2>=pi - pup_ang,phi2<=pi + pup_ang];

    poliABC = [1, 2*(ex*x0 + ey*y0 + ez*z0), x0^2 + y0^2 + z0^2 - R^2];
    a_an = min(roots(poliABC));


    if isreal(a_an) % Does the ray hit the sphere?
        % X_1 -> point on sphere in first ray intersection
        X_1 = X_0 + a_an*e_0; % interface 1 location


        en_1 = X_1 / norm(X_1);%[X_1(1),X_1(2),X_1(3)]/R; % norm of incoming beam interface 1
        % pupil_loc = [floor((en_1(2)/2+0.5)*N_pup_dir+1), round((en_1(3)/2+0.5)*N_pup_dir+1)]; % location on the hemispherical face of the sphere for mask testing
        temp_theta = asin(X_1(2)/R);
        ind_x = round((temp_theta/pi+0.5)*(N_pup_dir-1)+1);
        temp_phi = asin(X_1(3)/R/cos(temp_theta));
        ind_y = round((temp_phi/pi+0.5)*(N_pup_dir-1)+1);

        if pupil_binary(ind_y,ind_x) % Does the ray cross the pupil?
            % Senll's law on first interface
            beta_outside = acos(-dot(en_1,e_0));
            beta_inside = asin(n1/n2*sin(beta_outside)); % refraction angle
            e_1 = n1/n2 * e_0 + (n1/n2 * cos(beta_outside) - cos(beta_inside)) * en_1; % - for n1/n2=1 we must get e_1 = e_0 <<<<<<<<<<<<<<<<<<<<<<
            e_1 = e_1/norm(e_1);

            % start working on graded index ray bending
            e_in = e_1;
            X_a = X_1;
            ra = sum(X_a.^2)^0.5;%(xa^2 + ya^2 + za^2)^0.5;
            na = n2_func(end);

            X_b = X_a + dl*e_1;
            rb = norm(X_b);%sum(X_b.^2)^0.5;%(xb^2+zb^2+yb^2)^0.5;
            [~,II] = min((R-rb)^2);
            nb = n2_func(II);

            while rb<R
               
                % grad_n = (-2*etta*n_core/R^2) * (X_a + X_b).*0.5;
                mid_point = (X_a + X_b)/2;
                grad_n = - mid_point * (temp1 * n_core) / (temp1 + (ra + rb)/2)^2;
                e_in = e_in + dl * (2/(nb+na)) * grad_n;
                e_in = e_in / norm(e_in);

                X_a = X_b;
                X_b = X_b + dl * e_in;
              
                ra = rb;
                rb = norm(X_b);%(xb^2+zb^2+yb^2)^0.5;
                [~,II] = min((R-rb)^2);
                na = nb;
                nb = n2_func(II);

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
            
                % incident on retina
                Retina_image{angle_ind} = zeros(3,N_focal_surfaces);
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
                        Retina_image{angle_ind}(:,gi) = [Intensity,Ix,Iy];
                    end
                end

       
            end

        end
    end
end

Retina_image_out = zeros(N_retina_angs,N_retina_angs,N_focal_surfaces);
for ni = 1:N_rays
    if ~isempty(Retina_image{ni,1})
        for gi = 1:N_focal_surfaces
            Retina_image_out(Retina_image{ni}(2,gi),Retina_image{ni}(3,gi),gi) = Retina_image_out(Retina_image{ni}(2,gi),Retina_image{ni}(3,gi),gi) + Retina_image{ni}(1,gi);
        end
    end
end

for si = 1:N_focal_surfaces
    Retina_image_out(:,:,si) = conv2(Retina_image_out(:,:,si),ker,'same');
end



clear all
% close all


% % this section creates a refractive index function of wavelength according
% % to some data from Jagger[1999] and a graded index function we invented
% % n(r) = n_core * (1 - etta * r^2 / R^2)

% wavelength = [450,500,550,600,650,700];
% n_core_vec = [1.513,1.507,1.502,1.498,1.496,1.494];
% n_medium_vec = [1.347,1.343,1.341,1.339,1.338,1.337];

% % normalize to account for refractive index media n0 = 1
% n_core_vec_norm = n_core_vec./n_medium_vec;
% aa = (n_core_vec_norm(2)-n_core_vec_norm(1))/50;
% 
% wavelength_vec = [449:451,499:501,549:551,599:601,649:651,699:701];%[450:50:700];
% n_core_all =  interp1([440,wavelength],[n_core_vec_norm(1)-aa*10,n_core_vec_norm],wavelength_vec,'spline');
% % plot(wavelength,n_core_vec_norm,wavelength_vec,n_core_all)
% % xlabel('wavelength [nm]')
% % ylabel('refractive index normalized to medium')

% --- Wavelengths of interest (we only use the ones from 450 to 700 nm) ---
wavelengths = [450, 550, 600, 700];

% --- Measured lens-center and medium indices at these wavelengths ---
% n_core_vec: index at lens center (absolute)
% n_medium_vec: index of surrounding medium (absolute)
%%%%%%%%%%%%%%%%%%%%%%%%%% FROM JAGGER ET. AL 1999
n_core_vec   = [1.62, 1.55, 1.52, 1.48]; % 450,550,600,700
n_medium_vec = [1.35, 1.33, 1.32, 1.305]; % 450,550,600,700


% Normalize and Interpolate the Lens-Center Indices
n_core_vec_norm = n_core_vec ./ n_medium_vec;   % ratio (center/medium)
n_external_vec_norm = (0.8*n_medium_vec + 0.2*n_core_vec)./n_medium_vec;
n_external_vec = (0.8*n_medium_vec + 0.2*n_core_vec);
wavelength_vec =  [450:5:700];

n_core_all = interp1(wavelengths, n_core_vec, wavelength_vec, 'spline');
n_external_all = interp1(wavelengths, n_external_vec, wavelength_vec, 'spline');
n_medium_all = interp1(wavelengths, n_medium_vec, wavelength_vec, 'spline');

figure;
% plot(wavelengths,n_core_vec,wavelengths,n_medium_vec)
plot(wavelength_vec,n_core_all,'LineWidth',2.5); hold on; grid on;
plot(wavelength_vec,n_external_all,'LineWidth',2.5)
plot(wavelength_vec,n_medium_all,'LineWidth',2.5)
legend('@ Sphere core','@ Outer-shell','Aquatic medium refractive index');
title('Refractive index spectral dependence')
xlabel('Wavelength [nm]')
ylabel('n(\lambda)')
xlim([wavelength_vec(1) wavelength_vec(end)])
ylim([1 1.7])
set(gca,'FontSize',14)


etta = 0.091;
R = 3;
r_vec = linspace(0,3,50);

figure(70);
cmap = colormap('turbo');
inds = round(size(cmap,1)/(length(wavelength_vec))-1);
count = 0;
leg = [];
for lami = 1:length(wavelength_vec)

    count = count+1;
    % n2_func = n_core_all(lami)*(1-etta*r_vec.^2/R^2);
    % plot(r_vec,n2_func); hold on;
    etta = n_core_all(lami)/n_external_all(lami);
    n2_func = n_core_all(lami)*(1+(etta-1)*(r_vec/R).^2).^-1;
    plot(r_vec/R,n2_func,'Color',cmap(inds*count,:),'LineWidth',2.5); hold on; grid on;
    n_lam{lami} = [n_core_all(lami),n_external_all(lami),n_medium_all(lami)];
    leg = [leg,{['\lambda = ',num2str(wavelength_vec(lami)),'nm']}];
end
title('Graded index radial function')
xlabel('Normalized radius')
legend(leg)
set(gca,'FontSize',14)
ylabel('n(r)')
ylim([1 1.7])
%% 
pupil_file_path = 'full_app.png';

% ray_trace_sphere_3d([-95,20,-20],3,n_lam{17},5,20,pupil_file_path,false,true,50000,'r');
% ray_trace_sphere_3d([-80,0,0],3,n_lam{8},5,20,pupil_file_path,true,false,500,'g');
% ray_trace_sphere_3d([-110,-20,-20],3,n_lam{2},5,20,pupil_file_path,true,false,500,'b');

% ray_trace_sphere_3d([-200,0,0],3,n_lam{4},5,20,pupil_file_path,true,false,500,'y');
% ray_trace_sphere_3d([-200,0,0],3,n_lam{5},5,20,pupil_file_path,true,false,500,[0.9290 0.6940 0.1250]);
% ray_trace_sphere_3d([-200,0,0],3,n_lam{6},5,20,pupil_file_path,true,false,500,'r');
% ray_trace_sphere_3d([-200,0,0],3,n_lam{7},5,20,pupil_file_path,true,false,500,'k');
% 


%%

output_path = 'Full_new';
mkdir(output_path);


y_vec = 0;%0:0.2:10;
z_vec = -200;
x_vec = 0;%45:60;%1:1:15;% 30:1:40


Lens_radius = 3;
Retina_min = 5;
Retina_max = 20;
N_rays = 5e5;

for lami = 1:length(wavelength_vec)
    for yi =1:length(y_vec)
        for xi = 1:length(x_vec)
            for zi = 1:length(z_vec)
                Source_location = [z_vec(zi),x_vec(xi),y_vec(yi)];
                n_core = n_core_all(lami);
                out_file_name = [output_path,'\Retina_Dcuttlefish_z',num2str(-z_vec(zi)),'_x',num2str(x_vec(xi)),'_y',num2str(y_vec(yi)),'_lam',num2str(wavelength_vec(lami)),'.mat'];
                
                % % Option 1: with ploting - no parallel compute
                % do_3d_plot = false;
                % do_intensity_plot = true;
                % [retina_image,Retina_ang_ind,Retina_plane_vec] = ray_trace_sphere_3d(Source_location,Lens_radius,n_lam{lami},Retina_min,Retina_max,pupil_file_path,do_3d_plot,do_intensity_plot,N_rays,'k');
                
                % Option 2: without ploting and parallel compute
                [retina_image,Retina_ang_ind,Retina_plane_vec] = ray_trace_sphere_3d_par(Source_location,Lens_radius,n_lam{lami},Retina_min,Retina_max,pupil_file_path,N_rays);

                save(out_file_name,'retina_image','Retina_ang_ind','Retina_plane_vec');
            end
        end
    end
end
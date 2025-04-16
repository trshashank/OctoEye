%% plot wavelength on axis
output_path1 = 'Dwarf_Cuttlefish_narrow';
output_path2 = 'Dwarf_Cuttlefish_wide';

% v = VideoWriter("some_colours_moving_across_retina.mp4");
% v.FrameRate = 3;
% open(v)

wavelength_vec = [450:5:700];
load('fish_spectra_data.mat');
wavelength_weight1 = spectral_sig_1;%spectral_sig_3;
wavelength_weight2 = spectral_sig_2;%spectral_sig_4;

x0 = [0:15,45:60];
y0 = 0;
z0 = -200;

% leg = [];
figure(5);
% set(gcf,"Position",[0,0,1500,750])
colormap gray
subplot(2,5,1);
imagesc(imread("D_cuttlefish_narrow.png")); axis off
title('Narrow slit');
% subplot(2,4,5);
% imagesc(imread("D_cuttlefish.png")); axis off
subplot(2,5,6);
imagesc(imread("D_cuttlefish_wide.png")); axis off
title('Wide slit');

retina_y_lim = 250:501;
retina_x_lim = 150:601;

retina_depth1 = 20;
retina_depth2 = 20;
k=1
lami=1;
out_file_name = [output_path1,'\Retina_Dcuttlefish_z',num2str(-z0),'_x',num2str(x0(k)),'_y',num2str(y0),'_lam',num2str(wavelength_vec(lami)),'.mat'];
load(out_file_name);
retina_image_narrow_full1 = zeros(size(retina_image));
retina_image_narrow_full2 = zeros(size(retina_image));
retina_image_wide_full1 = zeros(size(retina_image));
retina_image_wide_full2 = zeros(size(retina_image));
% for k = 1:length(x0)
if x0(k)>=45
    y0 = 10;
else
    y0 = 0;
end

im1 = zeros(751,751);
im2 = zeros(751,751);
for lami = 1:length(wavelength_vec)
    out_file_name = [output_path1,'\Retina_Dcuttlefish_z',num2str(-z0),'_x',num2str(x0(k)),'_y',num2str(y0),'_lam',num2str(wavelength_vec(lami)),'.mat'];
    load(out_file_name);
    im1 = im1 + wavelength_weight1(lami) * squeeze(retina_image(:,:,retina_depth1)');
    im2 = im2 + wavelength_weight2(lami) * squeeze(retina_image(:,:,retina_depth2)');

    retina_image_narrow_full1 = retina_image_narrow_full1 + wavelength_weight1(lami) * retina_image;
    retina_image_narrow_full2 = retina_image_narrow_full2 + wavelength_weight2(lami) * retina_image;
end
subplot(2,5,[2,3]);
imagesc(Retina_ang_ind(retina_x_lim),Retina_ang_ind(retina_y_lim),im1(retina_y_lim,retina_x_lim));
subplot(2,5,[4,5]);
imagesc(Retina_ang_ind(retina_x_lim),Retina_ang_ind(retina_y_lim),im2(retina_y_lim,retina_x_lim));

im1 = zeros(751,751);
im2 = zeros(751,751);
for lami = 1:length(wavelength_vec)
    out_file_name = [output_path2,'\Retina_Dcuttlefish_z',num2str(-z0),'_x',num2str(x0(k)),'_y',num2str(y0),'_lam',num2str(wavelength_vec(lami)),'.mat'];
    load(out_file_name);
    im1 = im1 + wavelength_weight1(lami) * squeeze(retina_image(:,:,retina_depth1)');
    im2 = im2 + wavelength_weight2(lami) * squeeze(retina_image(:,:,retina_depth2)');

    retina_image_wide_full1 = retina_image_wide_full1 + wavelength_weight1(lami) * retina_image;
    retina_image_wide_full2 = retina_image_wide_full2 + wavelength_weight2(lami) * retina_image;
end
subplot(2,5,[7,8]);
imagesc(Retina_ang_ind(retina_x_lim),Retina_ang_ind(retina_y_lim),im1(retina_y_lim,retina_x_lim));
xlabel('Retina angle [rad]')
subplot(2,5,[9,10]);
imagesc(Retina_ang_ind(retina_x_lim),Retina_ang_ind(retina_y_lim),im2(retina_y_lim,retina_x_lim));
xlabel('Retina angle [rad]')
% title('Wide slit'); colorbar;

%%
figure; colormap gray;
diffw1 = diff(log(1+retina_image_wide_full1),1,3);
diffw2 = diff(log(1+retina_image_wide_full2),1,3);
diffn1 = diff(log(1+retina_image_narrow_full1),1,3);
diffn2 = diff(log(1+retina_image_narrow_full2),1,3);


for ri = 1:99
    subplot(2,2,1)
    imagesc(diffw1(retina_y_lim,retina_x_lim,ri)'); %colorbar
    title('Wide sig1')
    axis off

    subplot(2,2,2)
    imagesc(diffw2(retina_y_lim,retina_x_lim,ri)'); %colorbar
    title('Wide sig2')
    axis off

    subplot(2,2,3)
    imagesc(diffn1(retina_y_lim,retina_x_lim,ri)'); %colorbar
    title('Narrow sig1')
    axis off

    subplot(2,2,4)
    imagesc(diffn2(retina_y_lim,retina_x_lim,ri)'); %colorbar
    title('Narrow sig2')
    axis off

    drawnow;
    pause(0.1)


    % figure(6); % used for sampling the cross section of a specific line in the image 
    % temp = diffn2(300,retina_x_lim,ri);
    % temp2 = diffn1(300,retina_x_lim,ri);
    % 
    % plot((flip(temp)),'LineWidth',2.5,'Color','k'); hold on;
    % plot((flip(temp2)),'LineWidth',2.5,'LineStyle','--','Color','k');

end
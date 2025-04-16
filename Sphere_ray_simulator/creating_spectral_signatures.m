clear all
close all

% This scrip generates a set of arbitrary spectrums for testing 
% resolvability of continuous spectral signatures. The results are saved
% and used in simulation post-processing to asses the image created by 
% various colours in the scene.
% It also creates an image of a striped fish containing these signatures,
% but simulating a full multi-signature image is not included in this
% version of the code.

wavelength_vec =  450:5:700;
spectral_sig_bg = (cos((wavelength_vec-450)/95)+1)/2;
spectral_sig_bg(1:4) = spectral_sig_bg(1:4).*[0.2,0.5,0.8,0.95];

spectral_sig_1 = (sin((wavelength_vec-450)/25)+1)/4+(sin((wavelength_vec-450)/25)+1)/4;
spectral_sig_1(wavelength_vec<570) = 0;
spectral_sig_1(1:4) = spectral_sig_1(1:4).*[0.2,0.5,0.8,0.95];

spectral_sig_2 = (sin((wavelength_vec-380)/30)+1)/4+(cos((wavelength_vec-420)/35)+1)/4;
spectral_sig_2(1:4) = spectral_sig_2(1:4).*[0.2,0.5,0.8,0.95];

spectral_sig_3 = (cos((wavelength_vec-460)/55)+1)/2;
spectral_sig_3(wavelength_vec>630) = 0;
spectral_sig_3(1:4) = spectral_sig_3(1:4).*[0.2,0.5,0.8,0.95];

spectral_sig_4 = (sin((wavelength_vec-300)/30)+1)/2;
spectral_sig_4(wavelength_vec>630) = 0;
spectral_sig_4 = spectral_sig_4*0.7 + 0.3;
spectral_sig_4(1:4) = spectral_sig_4(1:4).*[0.2,0.5,0.8,0.95];
spectral_sig_4((end-1):end) = spectral_sig_4((end-1):end).*[0.9,0.5];



spec_RGB = spectrumRGB(wavelength_vec);
% plot(wavelength_vec,spec_RGB(1,:,1),'r');hold on;
% plot(wavelength_vec,spec_RGB(1,:,2),'g');hold on;
% plot(wavelength_vec,spec_RGB(1,:,3),'b');hold on;
figure; 

a = spectral_sig_bg;
r_bg = sum(a.*spec_RGB(1,:,1))/sum(a);
g_bg = sum(a.*spec_RGB(1,:,2))/sum(a);
b_bg = sum(a.*spec_RGB(1,:,3))/sum(a);
% plot(wavelength_vec,a,'Color',[r_bg,g_bg,b_bg],'LineWidth',2,'LineStyle','--'); hold on; grid on;

a = spectral_sig_1;
r_4 = sum(a.*spec_RGB(1,:,1))/sum(a);
g_4 = sum(a.*spec_RGB(1,:,2))/sum(a);
b_4 = sum(a.*spec_RGB(1,:,3))/sum(a);
plot(wavelength_vec,a,'Color',[r_4,g_4,b_4],'LineStyle','--','LineWidth',3); hold on; grid on;

a = spectral_sig_2;
r_2 = sum(a.*spec_RGB(1,:,1))/sum(a);
g_2 = sum(a.*spec_RGB(1,:,2))/sum(a);
b_2 = sum(a.*spec_RGB(1,:,3))/sum(a);
plot(wavelength_vec,a,'Color',[r_2,g_2,b_2],'LineWidth',3.5); 

% legend('sig bg','sig 1','sig 2')
legend('Spectrum 1','Spectrum 2')
set(gca,'FontSize',14)
xlabel('wavelength [nm]')
ylabel('normalized specra [a.u]')

figure;
plot(wavelength_vec,spectral_sig_bg,'Color',[r_bg,g_bg,b_bg],'LineWidth',2,'LineStyle','--'); hold on; grid on;

a = spectral_sig_3;
r_3 = sum(a.*spec_RGB(1,:,1))/sum(a);
g_3 = sum(a.*spec_RGB(1,:,2))/sum(a);
b_3 = sum(a.*spec_RGB(1,:,3))/sum(a);
plot(wavelength_vec,a,'Color',[r_3,g_3,b_3],'LineWidth',3); 

a = spectral_sig_4;
r_1 = sum(a.*spec_RGB(1,:,1))/sum(a);
g_1 = sum(a.*spec_RGB(1,:,2))/sum(a);
b_1 = sum(a.*spec_RGB(1,:,3))/sum(a);
plot(wavelength_vec,a,'Color',[r_1,g_1,b_1],'LineWidth',3.5); 

xlabel('wavelength [nm]')
ylabel('normalized specra [a.u]')
set(gca,'FontSize',14)
legend('sig bg','sig 3','sig 4')

%%

fish_im = double(imread('fish_image.bmp'));
fish_im(fish_im==15)=-1;
fish_im(fish_im<9 & fish_im>0)=2;
fish_im(fish_im>=9)=3;
fish_im(fish_im==-1)=1;
figure;
imagesc(fish_im)
left_ind = find(sum(fish_im,1)>size(fish_im,1),1);
right_ind = find(sum(fish_im,1)>size(fish_im,1),1,'last');
center_x = floor((right_ind-left_ind)/2 + left_ind);
bott_ind = find(sum(fish_im,2)>size(fish_im,2),1);
top_ind = find(sum(fish_im,2)>size(fish_im,2),1,'last');
center_y = floor(-(bott_ind-top_ind)/2 + bott_ind);

fish_im = fish_im(1:(2*center_y),1:(2*center_x));
imagesc(fish_im)
save("fish_spectra_data.mat",'fish_im','wavelength_vec','spectral_sig_bg','spectral_sig_1','spectral_sig_2','spectral_sig_3','spectral_sig_4')
%%







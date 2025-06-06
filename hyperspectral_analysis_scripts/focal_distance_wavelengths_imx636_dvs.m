parent_folder = "/media/samiarja/USB/OctoEye_paper_dataset/";
data = load(parent_folder+'imx-636-data/event_based_hyperspectral_results.mat');
addpath("hex2rgb.m")
range_plot = [2, 5];

wavelength = data.wavelength;
optimal_focal = data.optimal_focal;
segment_range = data.segment_range;  % 3-D array: [nRecords x 2 x 2]
event_rate_array = data.event_rate_array; % assumed to be a cell array of structs with fields 'timestamps' and 'values'

nRecords = numel(wavelength);

[sortedFocal, sortIdx] = sort(optimal_focal);
sortedWavelength = wavelength(sortIdx);
sortedSegmentRange = segment_range(sortIdx,:,:); % 3-D array sorted by optimal_focal
sortedERArray = event_rate_array(sortIdx);

% Initialize cell arrays to store (subset) focal and normalized event rate data
focal_cell = cell(nRecords,1);
evrate_cell = cell(nRecords,1);

colors = { '#610061', ... % 400nm
           'b',        ... % 450nm
           '#00ff92',  ... % 500nm
           'g',        ... % 550nm
           '#ffbe00',  ... % 600nm
           'r',        ... % 650nm
           '#e90000',  ... % 700nm
           '#a10000',  ... % 750nm
           '#6d0000',  ... % 800nm
           '#3b0f0f',  ... % 850nm
           '#210808',  ... % 900nm
           '#1c0404',  ... % 950nm
           '#030000'       % 1000nm
         };


range_plot = [3, 3.6];
final_focal = linspace(range_plot(1), range_plot(2), 50);  % 1D array of 50 points

% Preallocate the matrix for event rate curves (nRecords x 50)
peak_all_group = nan(nRecords, 50);
wavelength = double(wavelength);

hex_colors = { '#610061', '#0000FF', '#00ff92', '#00FF00', '#ffbe00', '#FF0000', ...
               '#e90000', '#a10000', '#6d0000', '#3b0f0f', '#210808', '#1c0404', '#030000' };

nColors = numel(hex_colors);
customRGB = zeros(nColors, 3);
for i = 1:nColors
    customRGB(i,:) = hex2rgb(hex_colors{i});
end

% Create an interpolated colormap for a continuous gradient.
color_wavelengths = double(wavelength);  
xi = linspace(min(color_wavelengths), max(color_wavelengths), 256);
interp_colormap = zeros(256,3);
for k = 1:3
    interp_colormap(:,k) = interp1(color_wavelengths, customRGB(:,k), xi, 'linear');
end

figure(567567);
clf;
set(gcf, 'Color', 'w', 'Position', [100 100 850 900]);
hold on;

% --- Create a smooth centerline from the optimal_focal data ---
t_fine = linspace(min(optimal_focal), max(optimal_focal), 2000);
w_fine = interp1(optimal_focal, double(wavelength), t_fine, 'linear'); 
% Use normalized wavelength for the gradient color
norm_w_fine = (w_fine - min(double(wavelength))) / (max(double(wavelength)) - min(double(wavelength)));

% --- Compute tangents and normals along the centerline ---
dx = gradient(t_fine);
dy = gradient(w_fine);
L = sqrt(dx.^2 + dy.^2);
% Normals (perpendicular to the tangent)
nx = -dy ./ L;
ny = dx ./ L;

% --- Define tube (thickened line) parameters ---
lineDiameter = 0.02;    % Total thickness (adjust as needed)
numCross = 15;           % Number of cross-section points across the tube

% Preallocate arrays for the tube mesh
offsets = linspace(-lineDiameter/2, lineDiameter/2, numCross);
X = zeros(numCross, length(t_fine));
Y = zeros(numCross, length(t_fine));

% Build the tube around the smooth centerline
for i = 1:length(t_fine)
    X(:, i) = t_fine(i) + offsets' * nx(i);
    Y(:, i) = w_fine(i) + offsets' * ny(i);
end

% Create a color matrix that varies along the centerline
C = repmat(norm_w_fine, numCross, 1);

% --- Plot the tube with interpolated colors and opacity ---
h = surf(X, Y, zeros(size(X)), C, 'EdgeColor', 'none', 'FaceColor', 'interp', 'FaceAlpha', 0.7);
colormap(interp_colormap);
shading interp;  % Smooth the color transition

% --- Overlay markers at each original data point
for i = 1:length(optimal_focal)
    scatter(optimal_focal(i), wavelength(i), 100, customRGB(i,:), 'filled', 'MarkerEdgeColor', 'k');
    
end

plot(optimal_focal, wavelength, '--k', 'LineWidth',2);

% --- Add a horizontal dashed line at 750 nm ---
yline(750, '--k', 'LineWidth', 2);

set(gca, 'YDir', 'reverse');  % Lower wavelengths appear at the top
set(gca, 'YTick', 400:50:1000, 'FontSize', 16, 'LineWidth', 3);
xlabel({'Distance from sensor to ball lens surface (cm)', '(Focal distance)'}, 'FontSize', 18, 'FontWeight', 'bold');ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
title("The change in focal distance with wavelength");

% --- Add rotated text annotations for spectral regions ---
text(3.175, 700, 'Visible light', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);
text(3.175, 950, 'Infrared', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);

grid on;
hold off;
xlim([3.163 3.4]);ylim([400 1000])


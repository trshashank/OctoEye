%% 13x13 grid for each wavelength and focal length - frames
parent_folder = "/media/samiarja/USB/OctoEye_paper_dataset/";
parent_path = parent_folder+'imx-249-data/';
addpath("hex2rgb.m")

pixel_intensity_limit = 35;
feedback_range_array = 2800:1:3250;


folders = {
    '400', ...
    '450', ...
    '500', ...
    '550', ...
    '600', ...
    '650', ...
    '700', ...
    '750', ...
    '800', ...
    '850', ...
    '900', ...
    '950', ...
    '1000',
    };


colors   = {
            '#610061', ... %400nm
            'b',...       % 450nm
            '#00ff92',... %500nm
            'g',...       %550nm
            '#ffbe00', ... %600nm
            'r',...       % 650nm
            '#e90000', ... %700nm
            '#a10000',...%750nm
            '#6d0000',...%800nm
            '#3b0f0f',...%850nm
            '#210808',...%900nm
            '#1c0404',...%950nm
            '#030000',%1000nm
            };      

% Feedback conversion parameters (not used in the subfigure display)
feedback_min = 284;
feedback_max = 3965;
distance_at_min_mm = 92;
feedback_range = feedback_max - feedback_min;

% Preallocate cell arrays for storing data from each folder.
nFolders = numel(folders);
focal_mm_all = cell(1, nFolders);
peak_all     = cell(1, nFolders);
fileListAll  = cell(1, nFolders);
maxFileIndex = cell(1, nFolders);

% Loop Over Each Folder to determine the file with the maximum corrected peak
vertCrop_half = 30;  % crop ±30 rows about the image center

feedback_motor_value = [];
frame_camera_hyperspectral_output = zeros(13,3);
for i = 1:nFolders
    currFolder = folders{i};
    fileList = dir(fullfile(parent_path, currFolder, '*.tiff'));
    fileListAll{i} = fileList;
    nFiles = numel(fileList);

    focal_vals = zeros(nFiles,1);
    peak_vals  = zeros(nFiles,1);
    
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    
    % Sort and reorder the struct array
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    
    % -- 2) Now, process each file in sorted order --
    for j = 1:nFiles
        fileName = fileList(j).name;
        [~, name] = fileparts(fileName);

        % feedback_value = str2double(name);
        feedback_value = feedback_range_array(j);
        % Read the image
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        
        img(img > pixel_intensity_limit) = 0;

        [rows, ~] = size(img);
        mid_row = round(rows/2);
        r_start = max(1, mid_row - 100);
        r_end   = min(rows, mid_row + 50);
        cropped = double(img(r_start:r_end, :));

        cropped_blurred = imgaussfilt(cropped, 2);

        % For each row in the cropped region, find its maximum pixel value.
        rowMaxima = max(cropped_blurred, [], 2);
        [peak, idx] = max(rowMaxima);
        
        % Apply QE correction.
        peak_vals(j) = peak;
        
    end
    
    % Find the file index with the maximum corrected peak.
    [peak_val, idx_max] = max(peak_vals);
    maxFileIndex{i} = idx_max;
    
    % (Storing focal length data omitted for brevity)
    % focal_mm_all{i} = [];  % not used below
    peak_all{i}     = peak_vals;

    selected_focal_length = str2double(fileList(idx_max).name(1:4));

    focal_length_mm = (feedback_max - selected_focal_length) / feedback_range * distance_at_min_mm;

    frame_camera_hyperspectral_output(i,:) = [str2double(folders{i}), focal_length_mm/10, peak_val];
    fprintf('Wavelength (nm) %s: Focal length (cm) %d (%s)\n', folders{i}, idx_max, focal_length_mm/10);
    feedback_motor_value = [feedback_motor_value;feedback_range_array(idx_max)];
end

% Parameters for the ROI (circle) extraction in each image:
ROI_size = 100;        % Use a 30x30 region (centered on the image center) to search for the brightest pixel.
ROI_half = ROI_size / 2;
crop_radius = 35;     % Final crop: (2*crop_radius+1) x (2*crop_radius+1) region around the detected peak.

% Subplot parameters for subtightplot:
gap    = [0.005 0.005];  % [vertical_gap, horizontal_gap]
marg_h = [0.02 0.02];    % [bottom_margin, top_margin]
marg_w = [0.02 0.02];    % [left_margin, right_margin]

% Create a 12x12 figure using subtightplot.
figure(1001); clf;
set(gcf, 'Color', 'w');


for row = 1:nFolders
    % For the "base" folder in this row, use its max-peak file index.
    base_idx = maxFileIndex{row};
    
    for col = 1:nFolders
        currFolder = folders{col};
        fileList = fileListAll{col};
        nFiles = numel(fileList);
        if base_idx > nFiles
            fileIdx = nFiles;
        else
            fileIdx = base_idx;
        end
        
        % Sort fileList based on numeric value in filenames.
        numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
        [~, sortIdx] = sort(numbers);
        fileList = fileList(sortIdx);

        fileName = fileList(fileIdx).name;
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3)==3
            img = rgb2gray(img);
        end
        
        % Convert image to double and apply QE correction for the current folder.
        img = double(img);
        [r, c] = size(img);
        center_y = round(r/2);
        center_x = round(c/2);
        
        % Define the 30x30 ROI about the image center.
        roi_y_min = center_y - ROI_half + 1;
        roi_y_max = center_y + ROI_half;
        roi_x_min = center_x - ROI_half + 1;
        roi_x_max = center_x + ROI_half;
        roi_y_min = max(1, roi_y_min);
        roi_y_max = min(r, roi_y_max);
        roi_x_min = max(1, roi_x_min); 
        roi_x_max = min(c, roi_x_max);
        ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
        
        % Smooth the ROI to mitigate the effect of hot pixels.
        % Adjust the sigma parameter as needed.
        ROI_smoothed = imgaussfilt(ROI_img, 3);
        
        % Find the brightest pixel in the smoothed ROI.
        [~, idx_roi] = max(ROI_smoothed(:));
        [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_smoothed), idx_roi);
        peak_y = roi_y_min - 1 + peak_y_roi;
        peak_x = roi_x_min - 1 + peak_x_roi;
        
        % Define the final crop window around the detected peak.
        x_range = (peak_x - crop_radius):(peak_x + crop_radius);
        y_range = (peak_y - crop_radius):(peak_y + crop_radius);
        x_range = max(1, min(c, x_range));
        y_range = max(1, min(r, y_range));
        final_ROI = img(y_range, x_range);
        final_ROI(final_ROI < 0) = 0;
        final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
        
        % Compute the subplot index in the 12x12 grid.
        subplot_index = (row - 1) * nFolders + col;
        subtightplot(nFolders, nFolders, subplot_index, gap, marg_h, marg_w);
        imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
        % red_profile_3= imgaussfilt(red_profile_3, 1);
        
        % If this cell lies on the diagonal, add a thick colored boundary.
        if row == col
            hold on;
            rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                      'EdgeColor', colors{col}, 'LineWidth', 4);
            hold off;
        end
    end
end


wavelength = double(frame_camera_hyperspectral_output(:,1));
optimal_focal = double(frame_camera_hyperspectral_output(:,2));
[sortedFocal, ~] = sort(optimal_focal);
wavelength = double(wavelength);

hex_colors = { '#610061', '#0000FF', '#00ff92', '#00FF00', '#ffbe00', '#FF0000', ...
               '#e90000', '#a10000', '#6d0000', '#3b0f0f', '#210808', '#1c0404', '#030000' };

nColors = numel(hex_colors);
customRGB = zeros(nColors, 3);
for i = 1:nColors
    customRGB(i,:) = hex2rgb(hex_colors{i});
end

color_wavelengths = double(wavelength);  
xi = linspace(min(color_wavelengths), max(color_wavelengths), 256);
interp_colormap = zeros(256,3);
for k = 1:3
    interp_colormap(:,k) = interp1(color_wavelengths, customRGB(:,k), xi, 'linear');
end

figure(567567);clf;
set(gcf, 'Color', 'w', 'Position', [100 100 850 900]);
hold on;

t_fine = linspace(min(sortedFocal), max(sortedFocal), 2000);
w_fine = interp1(sortedFocal, wavelength, t_fine, 'linear'); 
norm_w_fine = (w_fine - min(wavelength)) / (max(wavelength) - min(wavelength));

% --- Compute tangents and normals along the centerline ---
dx = gradient(t_fine);
dy = gradient(w_fine);
L = sqrt(dx.^2 + dy.^2);
% Normals (perpendicular to the tangent)
nx = -dy ./ L;
ny = dx ./ L;

% --- Define tube (thickened line) parameters ---
lineDiameter = 0.03;    % Total thickness (adjust as needed)
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
    scatter(sortedFocal(i), wavelength(i), 100, customRGB(i,:), 'filled', 'MarkerEdgeColor', 'k');
    
end

plot(sortedFocal, wavelength, '--k', 'LineWidth',2);

% --- Add a horizontal dashed line at 750 nm ---
yline(750, '--k', 'LineWidth', 2);

set(gca, 'YDir', 'reverse');  % Lower wavelengths appear at the top
set(gca, 'YTick', 400:50:1000, 'FontSize', 16, 'LineWidth', 3);
xlabel({'Distance from sensor to ball lens surface (cm)', '(Focal distance)'}, 'FontSize', 18, 'FontWeight', 'bold');ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
title("The change in focal distance with wavelength");

% --- Add rotated text annotations for spectral regions ---
text(2.28, 700, 'Visible light', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);
text(2.28, 950, 'Infrared', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);

grid on;
hold off;
xlim([2.26 2.625]);ylim([400 1000]);


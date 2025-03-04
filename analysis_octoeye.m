%% Figure 1: RGB distribution
% quantum_efficiency_cam = [0.73, 0.71, 0.45];
pixel_intensity_limit = 1100;
quantum_efficiency_cam = [1,1,1];
blue_qe  = quantum_efficiency_cam(1);
green_qe = quantum_efficiency_cam(2);
red_qe   = quantum_efficiency_cam(3);

parent_path = '/media/samiarja/USB/Optical_characterisation/Hyperspectral_new/';

% Read images
red_focal_length_red_wavelength    = imread(parent_path + "650/image_30.tiff");
red_focal_length_green_wavelength  = imread(parent_path + "550/image_30.tiff");
red_focal_length_blue_wavelength   = imread(parent_path + "450/image_30.tiff");

green_focal_length_red_wavelength   = imread(parent_path + "650/image_33.tiff");
green_focal_length_green_wavelength = imread(parent_path + "550/image_33.tiff");
green_focal_length_blue_wavelength  = imread(parent_path + "450/image_33.tiff");

blue_focal_length_red_wavelength    = imread(parent_path + "650/image_37.tiff");
blue_focal_length_green_wavelength  = imread(parent_path + "550/image_37.tiff");
blue_focal_length_blue_wavelength   = imread(parent_path + "450/image_37.tiff");

% Define subplot parameters for subtightplot (adjust as desired)
gap    = [0.01 0.01];    % [vertical_gap, horizontal_gap]
marg_h = [0.05 0.05];    % [bottom_margin, top_margin]
marg_w = [0.05 0.05];    % [left_margin, right_margin]

% Define ROI and cropping parameters
ROI_size    = 200;           % ROI is 30×30 pixels (centered in the image)
ROI_half    = ROI_size / 2;   % equals 15
crop_radius = 15;            % Final crop: pixels from peak in each direction (resulting in a (2*crop_radius+1)×(2*crop_radius+1) window)

% --- Display Cropped Subfigure Images and Detect Peak Rows ---
% We will process each image, display its cropped ROI in one subplot,
% and store the detected peak row (the row containing the brightest pixel).

% (For clarity, we use variable names that encode:
%   [focal_set]_[wavelength], where focal_set: 
%       'r_' = red_focal_length (Group 1),
%       'g_' = green_focal_length (Group 2),
%       'b_' = blue_focal_length (Group 3);
%   and wavelength: 'r' (red), 'g' (green), 'b' (blue).)

figure(56657); clf;

% --- Group 3: Blue Focal Set (Subplots 1–3) ---
% Subplot 1: blue_focal_length_blue_wavelength (blue channel image)
subtightplot(3,3,1, gap, marg_h, marg_w);
img = double(blue_focal_length_blue_wavelength) / blue_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);
[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_bb = roi_y_min - 1 + peak_y_roi;  % store peak row for blue_focal_length_blue_wavelength
peak_x_bb = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_bb - crop_radius):(peak_x_bb + crop_radius);
y_range = (peak_y_bb - crop_radius):(peak_y_bb + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
hold on;
rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                      'EdgeColor', 'b', 'LineWidth', 4);
hold off;

% Subplot 2: blue_focal_length_green_wavelength (green channel image on blue focal set)
subtightplot(3,3,2, gap, marg_h, marg_w);
img = double(blue_focal_length_green_wavelength) / green_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_bg = roi_y_min - 1 + peak_y_roi;  % store peak row for blue_focal_length_green_wavelength
peak_x_bg = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_bg - crop_radius):(peak_x_bg + crop_radius);
y_range = (peak_y_bg - crop_radius):(peak_y_bg + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);

% Subplot 3: blue_focal_length_red_wavelength (red channel image on blue focal set)
subtightplot(3,3,3, gap, marg_h, marg_w);
img = double(blue_focal_length_red_wavelength) / red_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_br = roi_y_min - 1 + peak_y_roi;  % store peak row for blue_focal_length_red_wavelength
peak_x_br = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_br - crop_radius):(peak_x_br + crop_radius);
y_range = (peak_y_br - crop_radius):(peak_y_br + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);

% --- Group 2: Green Focal Set (Subplots 4–6) ---
% Subplot 4: green_focal_length_blue_wavelength (blue channel image on green focal set)
subtightplot(3,3,4, gap, marg_h, marg_w);
img = double(green_focal_length_blue_wavelength) / blue_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_gb = roi_y_min - 1 + peak_y_roi;  % store peak row for green_focal_length_blue_wavelength
peak_x_gb = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_gb - crop_radius):(peak_x_gb + crop_radius);
y_range = (peak_y_gb - crop_radius):(peak_y_gb + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);


% Subplot 5: green_focal_length_green_wavelength (green channel image on green focal set)
subtightplot(3,3,5, gap, marg_h, marg_w);
img = double(green_focal_length_green_wavelength) / green_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_gg = roi_y_min - 1 + peak_y_roi;  % store peak row for green_focal_length_green_wavelength
peak_x_gg = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_gg - crop_radius):(peak_x_gg + crop_radius);
y_range = (peak_y_gg - crop_radius):(peak_y_gg + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
hold on;
rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                      'EdgeColor', 'g', 'LineWidth', 4);
hold off;

% Subplot 6: green_focal_length_red_wavelength (red channel image on green focal set)
subtightplot(3,3,6, gap, marg_h, marg_w);
img = double(green_focal_length_red_wavelength) / red_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_gr = roi_y_min - 1 + peak_y_roi;  % store peak row for green_focal_length_red_wavelength
peak_x_gr = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_gr - crop_radius):(peak_x_gr + crop_radius);
y_range = (peak_y_gr - crop_radius):(peak_y_gr + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);

% --- Group 1: Red Focal Set (Subplots 7–9) ---
% Subplot 7: red_focal_length_blue_wavelength (blue channel image on red focal set)
subtightplot(3,3,7, gap, marg_h, marg_w);
img = double(red_focal_length_blue_wavelength) / blue_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_rb = roi_y_min - 1 + peak_y_roi;  % store peak row for red_focal_length_blue_wavelength
peak_x_rb = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_rb - crop_radius):(peak_x_rb + crop_radius);
y_range = (peak_y_rb - crop_radius):(peak_y_rb + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);

% Subplot 8: red_focal_length_green_wavelength (green channel image on red focal set)
subtightplot(3,3,8, gap, marg_h, marg_w);
img = double(red_focal_length_green_wavelength) / green_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_rg = roi_y_min - 1 + peak_y_roi;  % store peak row for red_focal_length_green_wavelength
peak_x_rg = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_rg - crop_radius):(peak_x_rg + crop_radius);
y_range = (peak_y_rg - crop_radius):(peak_y_rg + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);

% Subplot 9: red_focal_length_red_wavelength (red channel image on red focal set)
subtightplot(3,3,9, gap, marg_h, marg_w);
img = double(red_focal_length_red_wavelength) / red_qe;
[rows, cols] = size(img);
center_y = round(rows/2); center_x = round(cols/2);
roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
ROI_img = imgaussfilt(ROI_img, 3);

[~, idx] = max(ROI_img(:));
[peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
peak_y_rr = roi_y_min - 1 + peak_y_roi;  % store peak row for red_focal_length_red_wavelength
peak_x_rr = roi_x_min - 1 + peak_x_roi;
x_range = (peak_x_rr - crop_radius):(peak_x_rr + crop_radius);
y_range = (peak_y_rr - crop_radius):(peak_y_rr + crop_radius);
final_ROI = img(y_range, x_range);
final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
hold on;
rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                      'EdgeColor', 'r', 'LineWidth', 4);
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intensity Distribution Plot Using the Detected Peak Rows and Cropped x–Axis
% Instead of using a naive middle row, we now extract the row at the detected
% peak (using the stored peak row for each image) and then crop the row to only
% include 20 pixels to the left and 20 pixels to the right of the detected peak 
% column. This makes the distribution more visible.

% For each image we assume the detected peak x–coordinate was stored earlier:
%   For the Red Focal Set (Group 1):
%       red_focal_length_red_wavelength: peak_x_rr, peak_y_rr
%       red_focal_length_green_wavelength: peak_x_rg, peak_y_rg
%       red_focal_length_blue_wavelength: peak_x_rb, peak_y_rb
%
%   For the Green Focal Set (Group 2):
%       green_focal_length_red_wavelength: peak_x_gr, peak_y_gr
%       green_focal_length_green_wavelength: peak_x_gg, peak_y_gg
%       green_focal_length_blue_wavelength: peak_x_gb, peak_y_gb
%
%   For the Blue Focal Set (Group 3):
%       blue_focal_length_red_wavelength: peak_x_br, peak_y_br
%       blue_focal_length_green_wavelength: peak_x_bg, peak_y_bg
%       blue_focal_length_blue_wavelength: peak_x_bb, peak_y_bb
%
% (Make sure these variables were computed during your subfigure processing.)

cropX = 15;  % Number of pixels to the left and right of the detected peak to include

% Group 1: Red Focal Set (red_focal_length_* images)
red_profile_1   = double(red_focal_length_red_wavelength(peak_y_rr, (peak_x_rr-cropX):(peak_x_rr+cropX)))   / red_qe;
green_profile_1 = double(red_focal_length_green_wavelength(peak_y_rg, (peak_x_rg-cropX):(peak_x_rg+cropX))) / green_qe;
blue_profile_1  = double(red_focal_length_blue_wavelength(peak_y_rb, (peak_x_rb-cropX):(peak_x_rb+cropX)))  / blue_qe;

red_profile_1= imgaussfilt(red_profile_1, 1);
green_profile_1= imgaussfilt(green_profile_1, 1);
blue_profile_1= imgaussfilt(blue_profile_1, 1);


% Group 2: Green Focal Set (green_focal_length_* images)
red_profile_2   = double(green_focal_length_red_wavelength(peak_y_gr, (peak_x_gr-cropX):(peak_x_gr+cropX)))   / red_qe;
green_profile_2 = double(green_focal_length_green_wavelength(peak_y_gg, (peak_x_gg-cropX):(peak_x_gg+cropX))) / green_qe;
blue_profile_2  = double(green_focal_length_blue_wavelength(peak_y_gb, (peak_x_gb-cropX):(peak_x_gb+cropX)))  / blue_qe;

red_profile_2= imgaussfilt(red_profile_2, 1);
green_profile_2= imgaussfilt(green_profile_2, 1);
blue_profile_2= imgaussfilt(blue_profile_2, 1);

% Group 3: Blue Focal Set (blue_focal_length_* images)
red_profile_3   = double(blue_focal_length_red_wavelength(peak_y_br, (peak_x_br-cropX):(peak_x_br+cropX)))   / red_qe;
green_profile_3 = double(blue_focal_length_green_wavelength(peak_y_bg, (peak_x_bg-cropX):(peak_x_bg+cropX))) / green_qe;
blue_profile_3  = double(blue_focal_length_blue_wavelength(peak_y_bb, (peak_x_bb-cropX):(peak_x_bb+cropX)))  / blue_qe;

red_profile_3= imgaussfilt(red_profile_3, 1);
green_profile_3= imgaussfilt(green_profile_3, 1);
blue_profile_3= imgaussfilt(blue_profile_3, 1);


% Because each cropped profile now has (2*cropX+1) columns, define a common x–axis:
x_axis = -cropX:cropX;  % This will be a vector of 41 points

% Define horizontal shifts to separate the groups in the plot
shift = 35;  % (adjust as needed)
x1 = x_axis;         % Group 3 (Blue Focal Set)
x2 = x_axis + shift; % Group 2 (Green Focal Set)
x3 = x_axis + 2*shift; % Group 1 (Red Focal Set)

figure(46567858); clf;
hold on;

% Enhance Axes Appearance
set(gca, 'LineWidth', 2, 'FontSize', 16);  % Thicker axes border & larger tick labels

% ----- Plot Group 3 (Blue Focal Set) using x1 -----
% For this first group, we store the line handles to use in the legend.
area(x1, red_profile_3,   'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
h_red = plot(x1, red_profile_3,   'r', 'LineWidth', 2.5);  % Store handle for legend

area(x1, green_profile_3, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
h_green = plot(x1, green_profile_3, 'g', 'LineWidth', 2.5);  % Store handle for legend

area(x1, blue_profile_3,  'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
h_blue = plot(x1, blue_profile_3,  'b', 'LineWidth', 2.5);  % Store handle for legend

% ----- Plot Group 2 (Green Focal Set) using x2 -----
area(x2, red_profile_2,   'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x2, red_profile_2,   'r', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x2, green_profile_2, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x2, green_profile_2, 'g', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x2, blue_profile_2,  'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x2, blue_profile_2,  'b', 'LineWidth', 2.5, 'HandleVisibility','off');

% ----- Plot Group 1 (Red Focal Set) using x3 -----
area(x3, red_profile_1,   'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x3, red_profile_1,   'r', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x3, green_profile_1, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x3, green_profile_1, 'g', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x3, blue_profile_1,  'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x3, blue_profile_1,  'b', 'LineWidth', 2.5, 'HandleVisibility','off');

% ----- Custom X-Ticks and Labels -----
% Calculate the center of each group for placing the ticks.
tick1 = mean(x1);
tick2 = mean(x2);
tick3 = mean(x3);
xticks([tick1, tick2, tick3]);
% xticklabels({'Blue Focal Length = 2.37cm', 'Green Focal Length = 2.47cm', 'Red Focal Length = 2.52cm'});
xticklabels({'Blue Focal Length', 'Green Focal Length', 'Red Focal Length'});

% ----- Axis Labels, Grid, and Legend -----
% xlabel('Pixel Column Relative to Peak', 'FontSize', 18);
ylabel('Intensity [a.u.]', 'FontSize', 18);
grid on;

% Only include the stored line handles in the legend (one per wavelength)
h_leg = legend([h_red, h_green, h_blue], {'650nm', '550nm', '450nm'}, 'Location', 'Best');
h_leg.Title.String = 'Wavelengths';
title("The profile of light source across the x-axis")

hold off;ylim([100 950]);


%% Greylevel hyperspectral analysis (old)
feedback_range_array = 2600:10:3400;
parent_path = '/media/samiarja/USB/Optical_characterisation/Hyperspectral_new/';
folders = {
    '400',...
    '450',...
    '500', ...
    '550',...
    '600',...
    '650',...
    '700',...
    '750',...
    '800',...
    '850',...
    '900',...
    '950',...
    '1000'
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


% Feedback conversion parameters
feedback_min = 284;       % At this raw feedback, camera is farthest away.
feedback_max = 3965;      % At this raw feedback, camera is closest.
distance_at_min_mm = 92;  % When feedback is at minimum, camera is 920 mm away.
feedback_range = feedback_max - feedback_min;

% Preallocate cell arrays for storing data from each folder.
focal_mm_all = cell(1, numel(folders));
peak_all     = cell(1, numel(folders));

% Loop Over Each Folder
for i = 1:numel(folders)
    currFolder = folders{i};
    fileList = dir(fullfile(parent_path, currFolder, '*.tiff'));
    nFiles = numel(fileList);
    
    % Preallocate arrays for the current folder.
    focal_vals = zeros(nFiles,1);  % focal lengths (in mm)
    peak_vals  = zeros(nFiles,1);  % corrected peak intensities
    
    % -- 1) Sort the fileList by the numeric part of the filename --
    % Extract the numeric portion from each file name using a regexp and convert to double
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    
    % Sort and reorder the struct array
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    
    % -- 2) Now, process each file in sorted order --
    for j = 1:nFiles
        fileName = fileList(j).name;
        [~, name] = fileparts(fileName);  % e.g. "image_3"
    

        % feedback_value = str2double(name);
        feedback_value = feedback_range_array(j);

        % Convert raw feedback to focal length (mm) using the dynamic formula:
        %   focal_length_mm = (feedback_max - feedback_value) / (feedback_max - feedback_min) * distance_at_min_mm
        focal_length_mm = (feedback_max - feedback_value) / feedback_range * distance_at_min_mm;
        
        % Read the image.
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end

        img(img > 65535) = 0;
        
        % Crop a vertical region around the image center (±30 rows)
        [rows, ~] = size(img);
        mid_row = round(rows/2);
        r_start = max(1, mid_row - 100);
        r_end   = min(rows, mid_row + 100);
        % cropped = double(img(r_start:r_end, :));

        cropped = double(img);

        % For each row in the cropped region, find its maximum pixel value.
        rowMaxima = max(cropped, [], 2);
        peak = max(rowMaxima);

        % figure(45654);imagesc(cropped);colorbar;title(fileName)
        
        % Apply the appropriate quantum efficiency correction.
        % Folder "450" uses blue (QE(1)), "550" uses green (QE(2)), "650" uses red (QE(3)).
        peak_corrected = peak;
        
        % Store the focal length and corrected peak value.
        focal_vals(j) = focal_length_mm;
        peak_vals(j)  = peak_corrected;
    end
    
    % Sort the data by focal length (x-axis)
    [focal_sorted, sortIdx] = sort(focal_vals);
    peak_sorted = peak_vals(sortIdx);
    
    focal_mm_all{i} = focal_sorted;
    peak_all{i}     = peak_sorted;
end

% Plotting: Combine All Datasets in a Visually Appealing Figure
figure(456457); clf; hold on;
set(gca, 'LineWidth', 2, 'FontSize', 16);  % Thicker axes and larger tick labels

% Preallocate handles for the legend (one per dataset)
h = gobjects(numel(folders),1);

for i = 1:numel(folders)
    % Convert focal lengths from mm to cm.
    x = focal_mm_all{i} / 10;  % x-axis: focal length in cm
    y = peak_all{i};           % y-axis: corrected peak intensity
    col = colors{i};

    
    
    % Plot a low-opacity shaded area under the curve.
    area(x, y, 'FaceColor', col, 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
    
    % Overlay the line with markers.
    h(i) = plot(x, y,'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8);
end

% Reverse the x-axis so that higher focal lengths (cm) appear on the left.
set(gca, 'XDir', 'reverse');

xlabel('Focal Length (cm)', 'FontSize', 24);
ylabel('Intensity', 'FontSize', 24);
grid on;
title("Frame Camera Hyperspectral Response")

% Rotate the x-axis tick labels by 45°.
xtickangle(45);

% Create a legend with only three entries (one per dataset).
legend(h, folders, 'Location', 'east', 'FontSize', 24);

% Add tags under the x-axis.
text(0, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'r');
text(1, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'r');

hold off;
xlim([2.2 2.9]);ylim([450 1100]);
% xlim([1.8 2.85]);ylim([3 21]);

%% Grelevel hyperspectral analysis (NEW!!!!!!!!!!!!!!!!!)
feedback_range_array = 2800:1:3250;
pixel_intensity_limit = 65535;
parent_path = '/media/samiarja/USB/Optical_characterisation/hyperspectral_high_resolution/';
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

% Feedback conversion parameters
feedback_min = 284;       % At this raw feedback, camera is farthest away.
feedback_max = 3965;      % At this raw feedback, camera is closest.
distance_at_min_mm = 92;  % When feedback is at minimum, camera is 920 mm away.
feedback_range = feedback_max - feedback_min;
% Preallocate cell arrays for storing data from each folder.
focal_mm_all = cell(1, numel(folders));
peak_all     = cell(1, numel(folders));
frame_camera_hyperspectral_output = zeros(13,2);

% Loop Over Each Folder
for i = 1:numel(folders)
    currFolder = folders{i};
    fileList = dir(fullfile(parent_path, currFolder, '*.tiff'));
    nFiles = numel(fileList);
    
    % Preallocate arrays for the current folder.
    focal_vals = zeros(nFiles,1);  % focal lengths (in mm)
    peak_vals  = zeros(nFiles,1);  % corrected peak intensities
    
    % -- 1) Sort the fileList by the numeric part of the filename --
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    
    % -- 2) Now, process each file in sorted order --
    for j = 1:nFiles
        fileName = fileList(j).name;
        [~, name] = fileparts(fileName);
    
        % feedback_value = str2double(name);
        feedback_value = feedback_range_array(j);
    
        % Convert raw feedback to focal length (mm) using the dynamic formula:
        %   focal_length_mm = (feedback_max - feedback_value) / (feedback_max - feedback_min) * distance_at_min_mm
        focal_length_mm = (feedback_max - feedback_value) / feedback_range * distance_at_min_mm;
        
        % Read the image.
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
    
        % Remove overly bright pixels.
        img(img > pixel_intensity_limit) = 0;
        
        % Crop a vertical region around the image center (from mid_row-100 to mid_row+50)
        [rows, ~] = size(img);
        mid_row = round(rows/2);
        r_start = max(1, mid_row - 100);
        r_end   = min(rows, mid_row + 50);
        cropped = double(img(r_start:r_end, :));
    
        % Apply Gaussian blur to the cropped image.
        cropped_blurred = imgaussfilt(cropped, 2);
    
        % Find the row with the maximum intensity in the blurred cropped image.
        rowMaxima = max(cropped_blurred, [], 2);
        [peak_corrected, peak_row] = max(rowMaxima);
    
        % In the row with the peak, find the column index of the maximum value.
        [~, peak_col] = max(cropped_blurred(peak_row, :));
        
        % Define the ROI: 15 pixels to the left/right and 15 pixels up/down around the peak.
        row_start_roi = max(1, peak_row - 15);
        row_end_roi   = min(size(cropped_blurred, 1), peak_row + 15);
        col_start_roi = max(1, peak_col - 15);
        col_end_roi   = min(size(cropped_blurred, 2), peak_col + 15);
    
        roi = cropped_blurred(row_start_roi:row_end_roi, col_start_roi:col_end_roi);
    
        % Divide the ROI into 9 patches (a 3x3 grid) and compute the maximum in each patch.
        [roi_rows, roi_cols] = size(roi);
        patch_row_size = floor(roi_rows / 3);
        patch_col_size = floor(roi_cols / 3);
        patch_max_values = zeros(3, 3);
    
        for p = 1:3
            for q = 1:3
                % Determine row indices for the patch.
                r1 = (p-1)*patch_row_size + 1;
                if p < 3
                    r2 = p*patch_row_size;
                else
                    r2 = roi_rows; % include any remaining rows in the last patch
                end
                
                % Determine column indices for the patch.
                c1 = (q-1)*patch_col_size + 1;
                if q < 3
                    c2 = q*patch_col_size;
                else
                    c2 = roi_cols; % include any remaining columns in the last patch
                end
                
                patch = roi(r1:r2, c1:c2);
                patch_max_values(p, q) = max(patch(:));
            end
        end
    
        % Calculate the corrected peak as the average of the maximums of the 9 patches.
        peak_corrected_by_mean = mean(patch_max_values(:));
    
        % Store the focal length and corrected peak value.
        focal_vals(j) = focal_length_mm;
        peak_vals(j)  = peak_corrected_by_mean; %peak_corrected;
    end
    
    % Sort the data by focal length (x-axis)
    [focal_sorted, sortIdx] = sort(focal_vals);
    peak_sorted = peak_vals(sortIdx);

    [f_peak_val, f_peak_idx] = max(peak_sorted);

    

    focal_selected = focal_sorted(f_peak_idx);

    
    focal_mm_all{i} = focal_sorted;
    peak_all{i}     = peak_sorted;

    frame_camera_hyperspectral_output(i,:) = [str2double(folders{i}), ...
        focal_selected/10,...
        ];

end

% Plotting: Combine All Datasets in a Visually Appealing Figure
figure(456457); clf; hold on;
set(gca, 'LineWidth', 2, 'FontSize', 16);  % Thicker axes and larger tick labels

% Preallocate handles for the legend (one per dataset)
h = gobjects(numel(folders),1);

for i = 1:numel(folders)
    % Convert focal lengths from mm to cm.
    x = focal_mm_all{i} / 10;  % x-axis: focal length in cm
    y = peak_all{i};           % y-axis: corrected peak intensity
    col = colors{i};

    windowSize = max(3, round(0.1 * numel(y)));
    y = smoothdata(y, 'movmean', windowSize);
    
    % Plot a low-opacity shaded area under the curve.
    area(x, y, 'FaceColor', col, 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
    
    % Overlay the line with markers.
    h(i) = plot(x, y,'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8);
end

% Reverse the x-axis so that higher focal lengths (cm) appear on the left.
% set(gca, 'XDir', 'reverse');

xlabel('Distance from sensor to ball lens surface (cm)', 'FontSize', 24);
ylabel('Maximum Intensity measured by the sensor [a.u.]', 'FontSize', 24);
grid on;title("Characterisation of the ball lens with a frame camera")

% Rotate the x-axis tick labels by 45°.
xtickangle(45);

% Create a legend with only three entries (one per dataset).
legend(h, folders, 'Location', 'east', 'FontSize', 24);

% % Add tags under the x-axis.
% text(0, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'r');
% text(1, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'r');


hold off;
xlim([1.9 2.9]);
ylim([2 22]);


figure(4564545);
clf;

% Loop over each dataset.
for i = 1:numel(folders)
    % if i == 1
    %     title("Frame Camera Hyperspectral Response", 'FontSize', 24);
    % end
    % Create a subplot (using subtightplot or subplot as available).
    subtightplot(numel(folders), 1, i);
    hold on;
    set(gca, 'LineWidth', 2, 'FontSize', 16);  % Thicker axes and larger tick labels
    
    % Convert focal lengths from mm to cm.
    x = focal_mm_all{i} / 10;  % x-axis: focal length in cm
    y = peak_all{i};           % y-axis: corrected peak intensity
    col = colors{i};

    windowSize = max(3, round(0.1 * numel(y)));
    y = smoothdata(y, 'movmean', windowSize);
    
    % Plot a low-opacity shaded area under the curve.
    area(x, y, 'FaceColor', col, 'FaceAlpha', 0.2, 'EdgeAlpha', 0, ...
         'HandleVisibility', 'off');
    
    % Overlay the line with markers.
    h = plot(x, y, 'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8);
    
    % Add a legend for this subplot (using the current dataset's label).
    legend(h, folders{i}, 'Location', 'west', 'FontSize', 24);
    
    xlim([2 2.8]);
    % 
    
    % Only show x-axis tick labels on the last subplot.
    if i < numel(folders)
        set(gca, 'XTickLabel', []);
    end

    % if i == 13
    %     xtickangle(45);
    %     xlabel('Focal Length (cm)', 'FontSize', 24);
    % end
    % if i == 7
    %     ylabel('Intensity', 'FontSize', 30);
    % end
    
    grid on;
    hold off;
    
    
    % if i == 13
    %     text(0, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    %         'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'r');
    %     text(1, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    %         'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'r');
    % end
end
ylim([3 22]);

% % Add tags under the x-axis in the last subplot.
% subplot(numel(folders), 1, numel(folders));
% frame_camera_hyperspectral_output = [wavelength wavelength];

%% Frame camera hyperspectral peaks and rainbow plot
load('/media/samiarja/USB/Optical_characterisation/hyperspectral_high_resolution/frame_camera_hyperspectral_output.mat');
addpath("hex2rgb.m")

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

figure(567567);
clf;
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
xlabel('Focal Length (cm)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
title("Frame Camera Hyperspectral Response");

% --- Add rotated text annotations for spectral regions ---
text(2.28, 700, 'Visible light', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);
text(2.28, 950, 'Infrared', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);

text(0, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 16, 'Color', 'r');
text(1, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 16, 'Color', 'r');

grid on;
hold off;
xlim([2.26 2.625]);ylim([400 1000])


%% 13x13 grid for each wavelength and focal length
pixel_intensity_limit = 35;

feedback_range_array = 2800:1:3250;
parent_path = '/media/samiarja/USB/Optical_characterisation/hyperspectral_high_resolution/';

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

% save("/media/samiarja/USB/Optical_characterisation/hyperspectral_high_resolution/frame_camera_hyperspectral_output.mat", frame_camera_hyperspectral_output);
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


%% ----------------------- Hyperspectral Rainbow Composite -----------------------

% SETTINGS & DEFINITIONS
feedback_range_array = 2800:1:3250;
parent_path = '/media/samiarja/USB/Optical_characterisation/hyperspectral_high_resolution/';

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



% Feedback conversion parameters (to compute focal length from filename feedback value)
feedback_min = 284;       % Minimum raw feedback (farthest)
feedback_max = 3965;      % Maximum raw feedback (closest)
distance_at_min_mm = 92;  % 920 mm when feedback is minimum
feedback_range = feedback_max - feedback_min;

% Preallocate cell arrays for storing each dataset.
nFolders = numel(folders);
focal_mm_all = cell(1, nFolders);  % each cell: vector of focal lengths (mm)
peak_all     = cell(1, nFolders);  % each cell: corresponding QE–corrected intensities

% ------------------ EXTRACT DATA FROM EACH FOLDER ------------------
for i = 1:nFolders
    currFolder = folders{i};
    fileList = dir(fullfile(parent_path, currFolder, '*.tiff'));
    nFiles = numel(fileList);
    
    focal_vals = zeros(nFiles,1);  % in mm
    peak_vals  = zeros(nFiles,1);  % QE–corrected peak intensity
    
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    
    % Sort and reorder the struct array
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    
    % -- 2) Now, process each file in sorted order --
    for j = 1:nFiles
        fileName = fileList(j).name;
        [~, name] = fileparts(fileName);  % e.g. "image_3"
    

        % feedback_value = str2double(name);
        feedback_value = feedback_range_array(j);
        
        % Convert raw feedback to focal length (mm)
        focal_length_mm = (feedback_max - feedback_value) / feedback_range * distance_at_min_mm;
        
        % Read image (convert to grayscale if needed)
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        
        % Crop a vertical region around the image center (±30 rows)
        img(img > pixel_intensity_limit) = 0;
        
        % Crop a vertical region around the image center (±30 rows)
        [rows, ~] = size(img);
        mid_row = round(rows/2);
        r_start = max(1, mid_row - 100);
        r_end   = min(rows, mid_row + 50);
        cropped = double(img(r_start:r_end, :));

        cropped_blurred = imgaussfilt(cropped, 2);


        % filteredImage = medfilt2(cropped, [5 5]);
        % maxIntensity = max(filteredImage(:));

        % figure(56757);imagesc(cropped_blurred);title(num2str(j))
        % cropped = double(img);

        % For each row in the cropped region, find its maximum pixel value.
        rowMaxima = max(cropped_blurred, [], 2);
        [peak, idx] = max(rowMaxima);
        
        % Apply the QE correction.
        peak_corrected = peak;
        
        focal_vals(j) = focal_length_mm;
        peak_vals(j) = peak_corrected;
    end
    
    % Sort each dataset by focal length.
    [focal_sorted, sortIdx] = sort(focal_vals);
    peak_sorted = peak_vals(sortIdx);
    
    focal_mm_all{i} = focal_sorted;
    peak_all{i} = peak_sorted;
end



% ------------------ COMPOSITE CURVE FROM STACKED (OVERLAPPING) DATASETS ------------------
% First, determine the overlapping x–range among all datasets.
x_min_all = zeros(1,nFolders);
x_max_all = zeros(1,nFolders);
for i = 1:nFolders
    x_i = focal_mm_all{i} / 10;
    x_min_all(i) = min(x_i);
    x_max_all(i) = max(x_i);
end
x_common_min = max(x_min_all);  % highest minimum
x_common_max = min(x_max_all);  % lowest maximum

% Define a common x–axis over the overlapping region.
nPoints = 500;
x_common = linspace(x_common_min, x_common_max, nPoints);

% Interpolate each dataset onto x_common.
y_interp = zeros(nFolders, nPoints);
for i = 1:nFolders
    x_i = focal_mm_all{i} / 10;
    y_i = peak_all{i};
    y_interp(i, :) = interp1(x_i, y_i, x_common, 'spline');
end

% For each x–value (i.e. each column), take the maximum over all datasets.
composite_y = max(y_interp, [], 1);  % composite envelope (one value per x)
% Also record which dataset gives the maximum at each x.
[~, idx_max] = max(y_interp, [], 1);

% ------------------ ASSIGN COLOR AT EACH X FROM THE WINNING CURVE ------------------
% For each x in x_common, assign the color corresponding to the dataset that produced the maximum.
discrete_colors = zeros(nPoints, 3);  % each row: [R G B]
for j = 1:nPoints
    dataset_idx = idx_max(j);
    % Convert the color string from colors{dataset_idx} to an RGB vector.
    colStr = colors{dataset_idx};
    if colStr(1) == '#'
        r_val = hex2dec(colStr(2:3));
        g_val = hex2dec(colStr(4:5));
        b_val = hex2dec(colStr(6:7));
        discrete_colors(j,:) = [r_val, g_val, b_val] / 255;
    else
        switch lower(colStr)
            case 'b', discrete_colors(j,:) = [0, 0, 1];
            case 'g', discrete_colors(j,:) = [0, 1, 0];
            case 'r', discrete_colors(j,:) = [1, 0, 0];
            case 'c', discrete_colors(j,:) = [0, 1, 1];
            case 'm', discrete_colors(j,:) = [1, 0, 1];
            case 'y', discrete_colors(j,:) = [1, 1, 0];
            otherwise, discrete_colors(j,:) = [0, 0, 0];
        end
    end
end

% Smooth the discrete color function using a moving average.
window = 15;  % adjust window size as needed
smooth_colors = zeros(size(discrete_colors));
for ch = 1:3
    smooth_colors(:,ch) = conv(discrete_colors(:,ch), ones(1,window)/window, 'same');
end

%
% Force the leftmost and rightmost 5% of x to black.
edge_fraction_left  = 0.25;
edge_fraction_right = 0.25;

nEdge_left = round(nPoints * edge_fraction_left);
nEdge_right = round(nPoints * edge_fraction_right);

smooth_colors(1:nEdge_left, :) = 0;
smooth_colors(end-nEdge_right+1:end, :) = 0;

% ------------------- PLOT COMPOSITE CURVE WITH GRADIENT FILL -------------------
figure(789); clf; hold on;
set(gca, 'FontSize', 16, 'LineWidth', 2);

% Plot the composite envelope as a thick black line.
plot(x_common, composite_y, 'k-', 'LineWidth', 4);

% To fill under the curve with a horizontal gradient (each vertical column having the same color):
% Build two-row matrices for x and y.
Xfill = repmat(x_common, 2, 1);         % 2 x nPoints
Yfill = [composite_y; zeros(1, nPoints)];  % top row is the composite curve; bottom row is 0
Zfill = zeros(2, nPoints);              % 2D surface (all zeros)

% For the fill colors, each column gets the corresponding smooth color (same for both rows).
% Create a 2 x nPoints x 3 array.
Cfill = zeros(2, nPoints, 3);
for j = 1:nPoints
    Cfill(:, j, :) = repmat(smooth_colors(j,:), 2, 1);
end

% Use surf to draw the filled area.
surf(Xfill, Yfill, Zfill, Cfill, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.5);

xlabel('Focal Length (cm)', 'FontSize', 24);
ylabel('Intensity', 'FontSize', 24);
title('Frame Camera Hyperspectral Response', 'FontSize', 28);

% Add tags under the x-axis.
text(0, -0.07, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'r');
text(1, -0.07, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'r');

% grid on;
hold off;
xlim([2.25 2.65]);
% ylim([550 1150]);
% set(gca, 'XDir', 'reverse');

%% ----------------------- 3D and 2D Intensity Map -----------------------
% Raw feedback range and conversion to focal length in cm
feedback_range_array = 2600:10:3400;  % raw feedback values
feedback_min = 284;       % Farthest (raw feedback)
feedback_max = 3965;      % Closest (raw feedback)
distance_at_min_mm = 92;  % 920 mm when feedback is minimum
feedback_range = feedback_max - feedback_min;

% Compute focal lengths (in mm) and convert to cm.
focal_length_mm = (feedback_max - feedback_range_array) / feedback_range * distance_at_min_mm;
focal_cm = focal_length_mm / 10;    % Now x-axis will be focal length in cm

% Define wavelengths (nm) from 400 to 1000 in 50 nm increments.
wavelengths = 400:50:1000;  % 13 discrete values
nWaves = numel(wavelengths);
nFocal = numel(feedback_range_array);

% (These folder names, colors, and QE factors should match your dataset.)
folders = {'400','450','500','550','600','650','700','750','800','850','900','950','1000'};
colors = {'#610061','b','#00ff92','g','#ffbe00','r','#e90000','#a10000','#6d0000','#3b0f0f','#210808','#1c0404','#030000'};
% QE_all = [0.95, 1, 1, 1, 1, 1, 1, 1, 0.99, 0.95, 1, 1, 0.7];
QE_all = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

pixel_intensity_limit = 60000;  % Upper limit for intensity (to reduce hot pixel effects)

% Preallocate intensity matrix (rows: wavelengths, columns: focal lengths)
intensity_mat = zeros(nWaves, nFocal);

% ------------------ LOOP OVER FOLDERS (Wavelengths) ------------------
% For each folder (each corresponding to a wavelength), loop over its images,
% extract the maximum intensity (after smoothing) and place it in the matrix.
for i = 1:nWaves
    currFolder = folders{i};
    fileList = dir(fullfile('D:\Optical_characterisation\Hyperspectral_new\', currFolder, '*.tiff'));
    
    % Sort fileList based on numeric value in filename.
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    
    % Loop over each image in the folder.
    for j = 1:length(fileList)
        fileName = fileList(j).name;
        fullFile = fullfile('D:\Optical_characterisation\Hyperspectral_new\', currFolder, fileName);
        img = imread(fullFile);
        if size(img,3)==3
            img = rgb2gray(img);
        end
        % Suppress outliers/hot pixels:
        img(img > pixel_intensity_limit) = 0;
        
        % Crop a vertical region about the image center (here ±300 rows for a larger ROI)
        [rows_img, ~] = size(img);
        mid_row = round(rows_img/2);
        r_start = max(1, mid_row - 300);
        r_end   = min(rows_img, mid_row + 300);
        cropped = double(img(r_start:r_end, :));

        
        % Smooth the cropped image to reduce hot pixel effects.
        img_smooth = imgaussfilt(cropped, 3);  % sigma=3; adjust as needed
        
        % Extract the maximum intensity from the smoothed image.
        max_intensity = max(img_smooth(:));
        
        % Extract the raw feedback value from the filename.
        numStr = regexp(fileName, '\d+', 'match', 'once');
        feedback_value = str2double(numStr);

        feedback_value = feedback_range_array(feedback_value+1);

        % Only process if feedback_value is within our desired range.
        if feedback_value < feedback_range_array(1) || feedback_value > feedback_range_array(end)
            continue;
        end
        
        % Determine the column index corresponding to this feedback value.
        col_idx = round((feedback_value - feedback_range_array(1)) / 10) + 1;
        
        % If multiple images map to the same focal value, keep the maximum intensity.
        intensity_mat(i, col_idx) = max(intensity_mat(i, col_idx), max_intensity);
    end
end

figure(567578);imagesc(intensity_mat);
xlabel("Focal length (cm)");
ylabel("Wavelengths (nm)");
% set(gca, 'YDir', 'reverse');


% ----------------------- 2D Intensity Map with Continuous Row-wise Color -----------------------

nWaves = 13; 
nFocal = numel(focal_cm); 
wavelengths = 400:50:1000;      % 400,450,...,1000 nm
focal_cm = linspace(2, 3, nFocal);  % Example focal lengths in cm

for i = 1:nWaves
    [~,peak_col] = max(intensity_mat(i,:));
    % intensity_mat(i,:) = intensity_mat(i,:);
    intensity_mat(i,:) = exp(-((1:nFocal)-peak_col).^2/10);
end

% --- Smooth and normalize the intensity matrix ---
smoothedIntensity = imgaussfilt(intensity_mat, 1);
normIntensity = smoothedIntensity / max(smoothedIntensity(:));

% --- Interpolate vertically to get a continuous y-axis ---
% Original vertical axis has 13 rows corresponding to wavelengths.
w_known = wavelengths;  
% Increase vertical resolution: use a number of points equal to the nm span.
nY = max(w_known) - min(w_known) + 1;  
y_interp_axis = linspace(min(w_known), max(w_known), nY);
% Interpolate the intensity matrix vertically (each column separately).
intensity_interp = interp1(w_known, normIntensity, y_interp_axis, 'spline', 'extrap');  % nY x nFocal

% --- Interpolate horizontally to get a continuous x-axis ---
% Increase horizontal resolution: here we multiply the number of focal points by 4.
nX = 2 * nFocal;
x_interp_axis = linspace(min(focal_cm), max(focal_cm), nX);
% Interpolate each row of intensity_interp horizontally.
intensity_interp_full = zeros(nY, nX);
for i = 1:nY
    intensity_interp_full(i,:) = interp1(focal_cm, intensity_interp(i,:), x_interp_axis, 'spline', 'extrap');
end

% --- Build the continuous colormap for the vertical direction ---
% Custom colors for each discrete wavelength (provided as hex or short names).
colorStrings = {'#610061','b','#00ff92','g','#ffbe00','r','#e90000','#a10000','#6d0000','#3b0f0f','#210808','#1c0404','#030000'};
customCmap = zeros(nWaves, 3);
for i = 1:nWaves
    colStr = colorStrings{i};
    if colStr(1)=='#'
        r_val = hex2dec(colStr(2:3));
        g_val = hex2dec(colStr(4:5));
        b_val = hex2dec(colStr(6:7));
        customCmap(i,:) = [r_val, g_val, b_val] / 255;
    else
        switch lower(colStr)
            case 'b', customCmap(i,:) = [0, 0, 1];
            case 'g', customCmap(i,:) = [0, 1, 0];
            case 'r', customCmap(i,:) = [1, 0, 0];
            otherwise, customCmap(i,:) = [0, 0, 0];
        end
    end
end

% Interpolate the custom colormap vertically to obtain a continuous colormap.
interpCmap = zeros(nY, 3);
for ch = 1:3
    interpCmap(:,ch) = interp1(w_known, customCmap(:,ch), y_interp_axis, 'spline', 'extrap');
end

% --- For IR wavelengths (>=780 nm), blend with grey so peaks remain visible ---
for i = 1:nY
    if y_interp_axis(i) >= 780
        blend = (y_interp_axis(i)-780) / (1000-780);  % blend factor from 0 at 780 nm to 1 at 1000 nm
        interpCmap(i,:) = (1-blend)*interpCmap(i,:) + blend*[0.5, 0.5, 0.5];
    end
end

% --- Build the final RGB image ---
% Each pixel's RGB = (normalized intensity^2) * (interpolated color for that row)
% We square the intensity to enhance the peak contrast.
RGB_img = zeros(nY, nX, 3);
for i = 1:nY
    for j = 1:nX
        brightness = intensity_interp_full(i,j)^2;
        RGB_img(i,j,:) = brightness * interpCmap(i,:);
    end
end

% ------------------ PLOT THE 2D INTENSITY MAP ------------------
figure(3001); clf;
% Display the RGB image with proper axis mapping:
imagesc(RGB_img, 'XData', x_interp_axis, 'YData', y_interp_axis);
axis on; 
text(0, -0.04, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', 'r');
text(1, -0.04, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'r');
xlabel('Focal Length (cm)', 'FontSize', 14);
ylabel('Wavelength (nm)', 'FontSize', 14);
% title('2D Intensity Map with Continuous Interpolation', 'FontSize', 18);

% Set x-axis to range from 2.3 to 2.55:
xlim([2.3 2.55]);
% Define x-ticks every 0.1 cm between 2.3 and 2.55.
set(gca, 'XDir', 'reverse');
xticks([]);

% Set y-axis ticks to display wavelengths from 400 to 1000 in 50 nm increments:
yticks(400:50:1000);
yticklabels(arrayfun(@(w) sprintf('%dnm', w), 400:50:1000, 'UniformOutput', false));


% ------------------ 3D PLOT: INTENSITY SURFACE with Continuous Row-wise Color ------------------
[nY, nX] = size(intensity_interp_full);

% Build the grid for the surface.
[X, Y] = meshgrid(x_interp_axis, y_interp_axis);
Z = intensity_interp_full;  % Use the full interpolated intensity matrix for z.

% Build an RGB array for the surface so that each row gets the corresponding
% interpolated color (and modulate brightness by squaring the intensity).
C3D = zeros(nY, nX, 3);
for i = 1:nY
    for j = 1:nX
        brightness = intensity_interp_full(i,j)^2;  % enhance peaks
        C3D(i,j,:) = brightness * interpCmap(i,:);
    end
end

figure(3002); clf; 
surf(X, Y, Z, C3D, 'EdgeColor', 'none');
xlabel('Focal Length (cm)', 'FontSize', 16);
ylabel('Wavelength (nm)', 'FontSize', 16);
zlabel('Intensity', 'FontSize', 16);

% title('3D Intensity Surface with Continuous Row-wise Color', 'FontSize', 18);

% Set the x-axis limits and flip the direction.
xlim([2.3 2.55]);
set(gca, 'XDir', 'reverse');


% Set y-ticks to display wavelengths from 400 to 1000 nm in 50 nm increments.
yticks(400:50:1000);
set(gca, 'YDir', 'normal');  % Ensure the y-axis is oriented normally.

% colorbar;
view(3);
set(gca, 'FontSize', 14, 'LineWidth', 1.5);

%% Longitudinal chromatic aberration
% observe the variation of the per-wavelength peak at different location
% quantum_efficiency_cam = [0.73, 0.71, 0.45];
parent_folder = "/media/samiarja/USB/Optical_characterisation/longitudinal_chromatic_aberration/";
pixel_intensity_limit = 1100;
smooth_factor = 5;
quantum_efficiency_cam = [1,1,1];
blue_qe  = quantum_efficiency_cam(1);
green_qe = quantum_efficiency_cam(2);
red_qe   = quantum_efficiency_cam(3);

point_location = 9;
parent_path = parent_folder + num2str(point_location) + "/";

if point_location == 1
    center_x = 1654; center_y = 383; %P1
end

if point_location == 2
    center_x = 1699; center_y = 523; %P2
end
if point_location == 3
    center_x = 1311; center_y = 411; %P3
end
if point_location == 4
    center_x = 958; center_y = 209;  %P4
end
if point_location == 5
    center_x = 1393; center_y = 195; %P5
end
if point_location == 6
    center_x = 1763; center_y = 177; %P6
end
if point_location == 7
    center_x = 1800; center_y = 65;  %P7
end
if point_location == 8
    center_x = 1500; center_y = 80;   %P8
end
if point_location == 9
    center_x = 1000; center_y = 95;   %P9
end



% Because each cropped profile now has (2*cropX+1) columns, define a common x–axis:
cropX = 15;  % Number of pixels to the left and right of the detected peak to include
x_axis = -cropX:cropX;  % This will be a vector of 41 points

% Define horizontal shifts to separate the groups in the plot
shift = 50;  % (adjust as needed)
x1 = x_axis;         % Group 3 (Blue Focal Set)
x2 = x_axis + shift; % Group 2 (Green Focal Set)
x3 = x_axis + 2*shift; % Group 1 (Red Focal Set)


% Read images
red_focal_length_red_wavelength    = imread(parent_path + "red_f_red_w.tiff");
red_focal_length_green_wavelength  = imread(parent_path + "red_f_green_w.tiff");
red_focal_length_blue_wavelength   = imread(parent_path + "red_f_blue_w.tiff");

green_focal_length_red_wavelength   = imread(parent_path + "green_f_red_w.tiff");
green_focal_length_green_wavelength = imread(parent_path + "green_f_green_w.tiff");
green_focal_length_blue_wavelength  = imread(parent_path + "green_f_blue_w.tiff");

blue_focal_length_red_wavelength    = imread(parent_path + "blue_f_red_w.tiff");
blue_focal_length_green_wavelength  = imread(parent_path + "blue_f_green_w.tiff");
blue_focal_length_blue_wavelength   = imread(parent_path + "blue_f_blue_w.tiff");


% Define subplot parameters for subtightplot (adjust as desired)
gap    = [0.01 0.01];    % [vertical_gap, horizontal_gap]
marg_h = [0.05 0.05];    % [bottom_margin, top_margin]
marg_w = [0.05 0.05];    % [left_margin, right_margin]

% Define ROI and cropping parameters
ROI_size    = 100;           % ROI is 30×30 pixels (centered in the image)
ROI_half    = ROI_size / 2;   % equals 15
crop_radius = 15;            % Final crop: pixels from peak in each direction (resulting in a (2*crop_radius+1)×(2*crop_radius+1) window)

figure(56657); clf;
if ~isempty(blue_focal_length_blue_wavelength)
    % --- Group 3: Blue Focal Set (Subplots 1–3) ---
    % Subplot 1: blue_focal_length_blue_wavelength (blue channel image)
    subtightplot(3,3,1, gap, marg_h, marg_w);
    img = double(blue_focal_length_blue_wavelength) / blue_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_bb = roi_y_min - 1 + peak_y_roi;  % store peak row for blue_focal_length_blue_wavelength
    peak_x_bb = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_bb - crop_radius):(peak_x_bb + crop_radius);
    y_range = (peak_y_bb - crop_radius):(peak_y_bb + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_blue_focal_length_blue_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
    hold on;
    rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                          'EdgeColor', 'b', 'LineWidth', 4);
    hold off;
end

if ~isempty(blue_focal_length_green_wavelength)
    % Subplot 2: blue_focal_length_green_wavelength (green channel image on blue focal set)
    subtightplot(3,3,2, gap, marg_h, marg_w);
    img = double(blue_focal_length_green_wavelength) / green_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_bg = roi_y_min - 1 + peak_y_roi;  % store peak row for blue_focal_length_green_wavelength
    peak_x_bg = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_bg - crop_radius):(peak_x_bg + crop_radius);
    y_range = (peak_y_bg - crop_radius):(peak_y_bg + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_blue_focal_length_green_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);    
end

if ~isempty(blue_focal_length_red_wavelength)
    % Subplot 3: blue_focal_length_red_wavelength (red channel image on blue focal set)
    subtightplot(3,3,3, gap, marg_h, marg_w);
    img = double(blue_focal_length_red_wavelength) / red_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_br = roi_y_min - 1 + peak_y_roi;  % store peak row for blue_focal_length_red_wavelength
    peak_x_br = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_br - crop_radius):(peak_x_br + crop_radius);
    y_range = (peak_y_br - crop_radius):(peak_y_br + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_blue_focal_length_red_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
end

if ~isempty(green_focal_length_blue_wavelength)
    % --- Group 2: Green Focal Set (Subplots 4–6) ---
    % Subplot 4: green_focal_length_blue_wavelength (blue channel image on green focal set)
    subtightplot(3,3,4, gap, marg_h, marg_w);
    img = double(green_focal_length_blue_wavelength) / blue_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_gb = roi_y_min - 1 + peak_y_roi;  % store peak row for green_focal_length_blue_wavelength
    peak_x_gb = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_gb - crop_radius):(peak_x_gb + crop_radius);
    y_range = (peak_y_gb - crop_radius):(peak_y_gb + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_green_focal_length_blue_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]); 
end

if ~isempty(green_focal_length_green_wavelength)
    % Subplot 5: green_focal_length_green_wavelength (green channel image on green focal set)
    subtightplot(3,3,5, gap, marg_h, marg_w);
    img = double(green_focal_length_green_wavelength) / green_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_gg = roi_y_min - 1 + peak_y_roi;  % store peak row for green_focal_length_green_wavelength
    peak_x_gg = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_gg - crop_radius):(peak_x_gg + crop_radius);
    y_range = (peak_y_gg - crop_radius):(peak_y_gg + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_green_focal_length_green_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
    hold on;
    rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                          'EdgeColor', 'g', 'LineWidth', 4);
    hold off; 
end

if ~isempty(green_focal_length_red_wavelength)
    % Subplot 6: green_focal_length_red_wavelength (red channel image on green focal set)
    subtightplot(3,3,6, gap, marg_h, marg_w);
    img = double(green_focal_length_red_wavelength) / red_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_gr = roi_y_min - 1 + peak_y_roi;  % store peak row for green_focal_length_red_wavelength
    peak_x_gr = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_gr - crop_radius):(peak_x_gr + crop_radius);
    y_range = (peak_y_gr - crop_radius):(peak_y_gr + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_green_focal_length_red_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
end


if ~isempty(red_focal_length_blue_wavelength)
    % --- Group 1: Red Focal Set (Subplots 7–9) ---
    % Subplot 7: red_focal_length_blue_wavelength (blue channel image on red focal set)
    subtightplot(3,3,7, gap, marg_h, marg_w);
    img = double(red_focal_length_blue_wavelength) / blue_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);

    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_rb = roi_y_min - 1 + peak_y_roi;  % store peak row for red_focal_length_blue_wavelength
    peak_x_rb = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_rb - crop_radius):(peak_x_rb + crop_radius);
    y_range = (peak_y_rb - crop_radius):(peak_y_rb + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_red_focal_length_blue_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
end

if ~isempty(red_focal_length_green_wavelength)
    % Subplot 8: red_focal_length_green_wavelength (green channel image on red focal set)
    subtightplot(3,3,8, gap, marg_h, marg_w);
    img = double(red_focal_length_green_wavelength) / green_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_rg = roi_y_min - 1 + peak_y_roi;  % store peak row for red_focal_length_green_wavelength
    peak_x_rg = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_rg - crop_radius):(peak_x_rg + crop_radius);
    y_range = (peak_y_rg - crop_radius):(peak_y_rg + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_red_focal_length_green_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
end

if ~isempty(red_focal_length_red_wavelength)
    % Subplot 9: red_focal_length_red_wavelength (red channel image on red focal set)
    subtightplot(3,3,9, gap, marg_h, marg_w);
    img = double(red_focal_length_red_wavelength) / red_qe;
    [rows, cols] = size(img);
    % center_y = round(rows/2); center_x = round(cols/2);
    % center_x = 1654; center_y = 383;
    roi_y_min = center_y - ROI_half + 1; roi_y_max = center_y + ROI_half;
    roi_x_min = center_x - ROI_half + 1; roi_x_max = center_x + ROI_half;
    ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
    ROI_img = imgaussfilt(ROI_img, smooth_factor);
    
    [~, idx] = max(ROI_img(:));
    [peak_y_roi, peak_x_roi] = ind2sub(size(ROI_img), idx);
    peak_y_rr = roi_y_min - 1 + peak_y_roi;  % store peak row for red_focal_length_red_wavelength
    peak_x_rr = roi_x_min - 1 + peak_x_roi;
    x_range = (peak_x_rr - crop_radius):(peak_x_rr + crop_radius);
    y_range = (peak_y_rr - crop_radius):(peak_y_rr + crop_radius);
    final_ROI = img(y_range, x_range);
    final_ROI(final_ROI < 0) = 0; final_ROI(final_ROI > pixel_intensity_limit) = pixel_intensity_limit;
    final_red_focal_length_red_wavelength = imgaussfilt(final_ROI, 1);
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
    hold on;
    rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                          'EdgeColor', 'r', 'LineWidth', 4);
    hold off;
end

point_focal_information = zeros(numel(x_axis),3);
image_information       = zeros(numel(x_axis),numel(x_axis),3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intensity Distribution Plot
% Group 1: Red Focal Set (red_focal_length_* images)
red_profile_1   = double(red_focal_length_red_wavelength(peak_y_rr, (peak_x_rr-cropX):(peak_x_rr+cropX)))   / red_qe;
green_profile_1 = double(red_focal_length_green_wavelength(peak_y_rg, (peak_x_rg-cropX):(peak_x_rg+cropX))) / green_qe;
blue_profile_1  = double(red_focal_length_blue_wavelength(peak_y_rb, (peak_x_rb-cropX):(peak_x_rb+cropX)))  / blue_qe;

red_profile_1   = imgaussfilt(red_profile_1, 1);
green_profile_1 = imgaussfilt(green_profile_1, 1);
blue_profile_1  = imgaussfilt(blue_profile_1, 1);
point_focal_information(:,1) = red_profile_1;
image_information(:,:,1) = final_red_focal_length_red_wavelength;

% Group 2: Green Focal Set (green_focal_length_* images)
red_profile_2   = double(green_focal_length_red_wavelength(peak_y_gr, (peak_x_gr-cropX):(peak_x_gr+cropX)))   / red_qe;
green_profile_2 = double(green_focal_length_green_wavelength(peak_y_gg, (peak_x_gg-cropX):(peak_x_gg+cropX))) / green_qe;
blue_profile_2  = double(green_focal_length_blue_wavelength(peak_y_gb, (peak_x_gb-cropX):(peak_x_gb+cropX)))  / blue_qe;

red_profile_2= imgaussfilt(red_profile_2, 1);
green_profile_2= imgaussfilt(green_profile_2, 1);
blue_profile_2= imgaussfilt(blue_profile_2, 1);
point_focal_information(:,2) = green_profile_2;
image_information(:,:,2) = final_green_focal_length_green_wavelength;

% Group 3: Blue Focal Set (blue_focal_length_* images)
red_profile_3   = double(blue_focal_length_red_wavelength(peak_y_br, (peak_x_br-cropX):(peak_x_br+cropX)))   / red_qe;
green_profile_3 = double(blue_focal_length_green_wavelength(peak_y_bg, (peak_x_bg-cropX):(peak_x_bg+cropX))) / green_qe;
blue_profile_3  = double(blue_focal_length_blue_wavelength(peak_y_bb, (peak_x_bb-cropX):(peak_x_bb+cropX)))  / blue_qe;

red_profile_3= imgaussfilt(red_profile_3, 1);
green_profile_3= imgaussfilt(green_profile_3, 1);
blue_profile_3= imgaussfilt(blue_profile_3, 1);
point_focal_information(:,3) = blue_profile_3;
image_information(:,:,3) = final_blue_focal_length_blue_wavelength;

figure(46567858); clf;
hold on;

% Enhance Axes Appearance
set(gca, 'LineWidth', 2, 'FontSize', 16);  % Thicker axes border & larger tick labels

% ----- Plot Group 3 (Blue Focal Set) using x1 -----
% For this first group, we store the line handles to use in the legend.
area(x1, red_profile_3,   'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
h_red = plot(x1, red_profile_3,   'r', 'LineWidth', 2.5);  % Store handle for legend

area(x1, green_profile_3, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
h_green = plot(x1, green_profile_3, 'g', 'LineWidth', 2.5);  % Store handle for legend

area(x1, blue_profile_3,  'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
h_blue = plot(x1, blue_profile_3,  'b', 'LineWidth', 2.5);  % Store handle for legend

% ----- Plot Group 2 (Green Focal Set) using x2 -----
area(x2, red_profile_2,   'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x2, red_profile_2,   'r', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x2, green_profile_2, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x2, green_profile_2, 'g', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x2, blue_profile_2,  'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x2, blue_profile_2,  'b', 'LineWidth', 2.5, 'HandleVisibility','off');

% ----- Plot Group 1 (Red Focal Set) using x3 -----
area(x3, red_profile_1,   'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x3, red_profile_1,   'r', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x3, green_profile_1, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x3, green_profile_1, 'g', 'LineWidth', 2.5, 'HandleVisibility','off');

area(x3, blue_profile_1,  'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
plot(x3, blue_profile_1,  'b', 'LineWidth', 2.5, 'HandleVisibility','off');

% ----- Custom X-Ticks and Labels -----
% Calculate the center of each group for placing the ticks.
tick1 = mean(x1);
tick2 = mean(x2);
tick3 = mean(x3);
xticks([tick1, tick2, tick3]);
xticklabels({'Blue Focal Length', 'Green Focal Length', 'Red Focal Length'});

% ----- Axis Labels, Grid, and Legend -----
% xlabel('Pixel Column Relative to Peak', 'FontSize', 18);
ylabel('Intensity', 'FontSize', 18);
grid on;

% Only include the stored line handles in the legend (one per wavelength)
legend([h_red, h_green, h_blue], {'650nm wavelength', '550nm wavelength', '450nm wavelength'}, 'Location', 'Best');

hold off;
% ylim([100 950]);

save(parent_folder+"point_focal_information_"+point_location+".mat","point_focal_information");
save(parent_folder+"image_information_"+point_location+".mat","image_information");

%% Longitudinal chromatic aberration plot
matFolder = '/media/samiarja/USB/Optical_characterisation/longitudinal_chromatic_aberration/';

% Get list of all .mat files matching your naming pattern
matFiles = dir(fullfile(matFolder, 'point_focal_information_*.mat'));
pixel_intensity_limit = 1100;
redProfiles   = [];
greenProfiles = [];
blueProfiles  = [];

% LOAD FILES AND ACCUMULATE DATA
for k = 1:length(matFiles)
    S = load(fullfile(matFolder, matFiles(k).name));
    fields = fieldnames(S);
    dataMatrix = S.(fields{1});
    
    % Check that the dataMatrix has at least 3 columns.
    if size(dataMatrix,2) < 3
        warning('File %s does not have at least 3 columns. Skipping.', matFiles(k).name);
        continue;
    end
    
    % Append each column (each column is a profile vector)
    redProfiles   = [redProfiles,   dataMatrix(:,1)];  %#ok<AGROW>
    greenProfiles = [greenProfiles, dataMatrix(:,2)];  %#ok<AGROW>
    blueProfiles  = [blueProfiles,  dataMatrix(:,3)];  %#ok<AGROW>
end

% COMPUTE AVERAGE AND STANDARD DEVIATION FOR EACH GROUP
% We assume all profiles have the same length.
red_mean   = mean(redProfiles, 2);
green_mean = mean(greenProfiles, 2);
blue_mean  = mean(blueProfiles, 2);

red_std   = std(redProfiles, 0, 2);
green_std = std(greenProfiles, 0, 2);
blue_std  = std(blueProfiles, 0, 2);

% DEFINE X-AXIS FOR EACH GROUP
% We want three separate segments along the x-axis:
% Left: Blue focal set (blue column), Middle: Green focal set, Right: Red focal set.
L = length(red_mean);    % Number of points in each profile
margin = 5;              % Gap (in x-units) between groups

x_blue  = 7 + (0:(L-1));                  % Blue group: x = 7,8,...,7+L-1
x_green = x_blue + L + margin;            % Green group shifted to the right
x_red   = x_blue + 2*(L + margin);          % Red group shifted further right

% Calculate the center positions for setting the x-ticks.
tick_blue  = mean(x_blue);
tick_green = mean(x_green);
tick_red   = mean(x_red);

% PLOT THE AVERAGE PROFILES WITH SHADING (PATCH for error region)
figure(46567858); clf;
% Position the plot using subtightplot in a 6x9 grid, occupying positions [1 23].
subtightplot(6,9,[1 22], [0.015 0.005], [0.12 0.02], [0.07 0.001]); 
hold on;
set(gca, 'LineWidth', 2, 'FontSize', 16);

% ----- Blue Group (Left Segment) -----
% Create a patch for mean ± std
X_patch = [x_blue, fliplr(x_blue)];
Y_patch = [blue_mean' - blue_std', fliplr(blue_mean' + blue_std')];
fill(X_patch, Y_patch, 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% Plot the average line on top
h_blue = plot(x_blue, blue_mean, 'b', 'LineWidth', 2.5);

% ----- Green Group (Middle Segment) -----
X_patch = [x_green, fliplr(x_green)];
Y_patch = [green_mean' - green_std', fliplr(green_mean' + green_std')];
fill(X_patch, Y_patch, 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h_green = plot(x_green, green_mean, 'g', 'LineWidth', 2.5);

% ----- Red Group (Right Segment) -----
X_patch = [x_red, fliplr(x_red)];
Y_patch = [red_mean' - red_std', fliplr(red_mean' + red_std')];
fill(X_patch, Y_patch, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h_red = plot(x_red, red_mean, 'r', 'LineWidth', 2.5);

% CUSTOMIZE AXES, TICKS, AND LEGEND
% Set custom x-ticks and labels for the three groups.
xticks([tick_blue, tick_green, tick_red]);
xticklabels({'Blue Focal Length', 'Green Focal Length', 'Red Focal Length'});
% Y-axis label
ylabel('Intensity', 'FontSize', 18);
grid on;
% Add a legend (the order here corresponds to: Red column = 650nm, Green column = 550nm, Blue column = 450nm)
% legend([h_red, h_green, h_blue], {'650nm wavelength', '550nm wavelength', '450nm wavelength'}, 'Location', 'Best');
hold off;
text(22-5,940,'450nm','Color','blue','FontSize',14)
text(58-5,940,'550nm','Color','green','FontSize',14)
text(94-5,940,'650nm','Color','red','FontSize',14)

matFiles = dir(fullfile(matFolder, 'image_information_*.mat'));

% LOAD FILES AND ACCUMULATE DATA
for k = 1:length(matFiles)
    S = load(fullfile(matFolder, matFiles(k).name));
    fields = fieldnames(S);
    dataMatrix = S.(fields{1});

    redProfiles   = dataMatrix(:,:,1);
    subtightplot(6,9,27+k, [0.005 0.005], [0.02 0.02], [0.02 0.02]);imshow(redProfiles, [0 pixel_intensity_limit]);
    hold on;text(5,5,num2str(k),'Color','red','FontSize',14)

    rectangle('Position',[0.5, 0.5, size(redProfiles,2), size(redProfiles,1)], ...
                          'EdgeColor', 'r', 'LineWidth', 4);hold off;
    greenProfiles = dataMatrix(:,:,2);
    subtightplot(6,9,36+k, [0.005 0.005], [0.02 0.02], [0.02 0.02]);imshow(redProfiles, [0 pixel_intensity_limit]);
    hold on;hold on;text(5,5,num2str(k),'Color','green','FontSize',14)
    rectangle('Position',[0.5, 0.5, size(greenProfiles,2), size(greenProfiles,1)], ...
                          'EdgeColor', 'g', 'LineWidth', 4);hold off;
    blueProfiles  = dataMatrix(:,:,3);
    subtightplot(6,9,45+k, [0.005 0.005], [0.02 0.02], [0.02 0.02]);imshow(redProfiles, [0 pixel_intensity_limit]);
    hold on;hold on;text(5,5,num2str(k),'Color','blue','FontSize',14)
    rectangle('Position',[0.5, 0.5, size(blueProfiles,2), size(blueProfiles,1)], ...
                          'EdgeColor', 'b', 'LineWidth', 4);hold off;
end

long_chroma_abb = imread("./figures/longidutinal_chroma.png");
subtightplot(6,9,[6 27], [0.01 0.01], [0.14 0.03], [-0.1 0.02])
imshow(imresize(long_chroma_abb,1))

%% Event based hyperspectral peaks and rainbow plot
data = load('/media/samiarja/USB/gen4-windows/recordings/event_based_hyperspectral_results.mat');
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

figure(34356);
hold on;
for i = 1:nRecords
    % For record i, extract the segment range.
    segStart = squeeze(sortedSegmentRange(i,:,1)); % [start_focal, start_time]
    segEnd   = squeeze(sortedSegmentRange(i,:,2)); % [end_focal, end_time]
    start_focal = segStart(1);
    start_time  = segStart(2);
    end_focal   = segEnd(1);
    end_time    = segEnd(2);
    
    % Get the event rate data for this record.
    er = sortedERArray{i};  % now using cell-array indexing (er.timestamps and er.values are numeric vectors)
    
    % Map event rate timestamps to focal length via linear interpolation.
    focal_vals = interp1([start_time, end_time], [start_focal, end_focal], er.timestamps);
    
    % Normalize the event rate values to [0, 1].
    er_min = min(er.values);
    er_max = max(er.values);
    norm_evrate_values = (er.values - er_min) / (er_max - er_min);
    
    % Select only the points where the focal value lies between 3 and 3.6 cm.
    mask = (focal_vals >= range_plot(1)) & (focal_vals <= range_plot(2));
    focal_subset = focal_vals(mask);
    evrate_subset = norm_evrate_values(mask);
    
    % Store this subset into cell arrays.
    focal_cell{i} = focal_subset;
    evrate_cell{i} = evrate_subset;
    
    % Also, plot the normalized curve for visual reference.
    if i <= numel(colors)
        plot(focal_subset, evrate_subset, 'Color', colors{i}, 'LineWidth', 2);
    else
        plot(focal_subset, evrate_subset, 'LineWidth', 2);
    end
end

xlabel('Focal Length (cm)');
ylabel('Normalized Event Rate (events/s)');
title('Normalized Event Rate Ranges (Focal: 3 to 3.6 cm)');
legendLabels = cell(nRecords,1);
for i = 1:nRecords
    legendLabels{i} = sprintf('%dnm (f = %.2f cm)', sortedWavelength(i), sortedFocal(i));
end
legend(legendLabels, 'Location', 'best');
xlim([3 3.6])
hold off;
grid on;
set(gca, 'XDir', 'reverse');

% Combine the extracted data into row-wise matrices
% First, determine the minimum length among all records.
minLen = min(cellfun(@length, focal_cell));

% Preallocate matrices (each row corresponds to one wavelength record)
focal_matrix = zeros(nRecords, minLen);
evrate_matrix = zeros(nRecords, minLen);

for i = 1:nRecords
    if i==1 || i ==7
        evrate_matrix(i,:) = flip(evrate_cell{i}(1:minLen));
        focal_matrix(i,:) = focal_cell{i}(1:minLen);
    else
        focal_matrix(i,:) = focal_cell{i}(1:minLen);
        evrate_matrix(i,:) = evrate_cell{i}(1:minLen); 
    end
end

% Ensure wavelength is double
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
norm_w_fine = (w_fine - min(wavelength)) / (max(wavelength) - min(wavelength));

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
xlabel('Focal Length (cm)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
title("Event Camera Hyperspectral Response");

% --- Add rotated text annotations for spectral regions ---
text(3.175, 700, 'Visible light', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);
text(3.175, 950, 'Infrared', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);

text(0, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 16, 'Color', 'r');
text(1, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 16, 'Color', 'r');

grid on;
hold off;
xlim([3.163 3.4]);ylim([400 1000])



%% 13x13 hyperspectral image grid
parent_path = '/media/samiarja/USB/Optical_characterisation/event_based_hyperspectral/';

folders = { '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000' };
colors = { '#610061', 'b', '#00ff92', 'g', '#ffbe00', 'r', '#e90000', '#a10000', '#6d0000', '#3b0f0f', '#210808', '#1c0404', '#030000' };

% Define custom file order (0-based indices) for each folder.
file_sequence = {
    [12 21 65 31 41 51 71 81 91 101 111 121], ...   % 400 nm
    [7 6 27 22 33 57 60 66 72 87 90 106], ...         % 450 nm
    [4 20 6 7 33 40 46 54 66 71 81 84], ...            % 500 nm
    [10 11 12 13 22 25 29 32 34 46 47 52], ...         % 550 nm
    [72 73 75 77 79 36 40 37 42 45 49 55], ...         % 600 nm
    [33 34 35 36 37 9 39 40 41 44 48 52], ...          % 650 nm
    [107 108 140 135 136 112 20 21 24 25 39 40], ...    % 700 nm
    [225 226 227 233 234 217 214 153 154 157 159 160], ... % 750 nm
    [195 192 191 183 184 185 160 112 37 38 70 71], ...  % 800 nm
    [85 78 77 72 70 69 68 66 7 6 12 16], ...           % 850 nm
    [66 58 56 55 42 41 40 39 37 22 21 25], ...         % 900 nm
    [79 72 58 57 55 42 41 40 38 9 8 7 1], ...          % 950 nm
    [82 76 75 67 60 59 58 19 14 13 12 10 9]            % 1000 nm
};

nFolders = numel(folders);
nFilesExpected = 13;  % Each folder is expected to have 13 images (indexed 0 to 12).

% Pre-read file lists for each folder.
fileListAll = cell(1, nFolders);
for col = 1:nFolders
    currFolder = folders{col};
    fileList = dir(fullfile(parent_path, currFolder, '*.png'));
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    fileListAll{col} = fileList;
end

% Define subplot spacing parameters for subtightplot.
gap    = [0.005 0.005];  % [vertical_gap, horizontal_gap]
marg_h = [0.02 0.02];    % [bottom_margin, top_margin]
marg_w = [0.02 0.02];    % [left_margin, right_margin]

% Create a 13x13 grid using subtightplot.
figure(1002); clf;
set(gcf, 'Color', 'w');

for row = 1:nFilesExpected   % row corresponds to focal index (0-based)
    for col = 1:nFolders    % col corresponds to wavelength folder
        % For folder 'col', get its file order vector.
        seq = file_sequence{col} + 1;  % convert from 0-based to MATLAB 1-based indexing
        fileList = fileListAll{col};
        nFiles = numel(fileList);
        if row > numel(seq)
            fileIdx = nFiles;
        else
            fileIdx = seq(row);
            if fileIdx > nFiles
                fileIdx = nFiles;
            end
        end
        
        % Load the image.
        fileName = fileList(fileIdx).name;
        img = imread(fullfile(parent_path, folders{col}, fileName));
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % Compute the subplot index.
        % Now, rows represent focal indices (from 0 to 12) and columns represent wavelengths (400 to 1000).
        subplot_index = (row - 1) * nFolders + col;
        subtightplot(nFilesExpected, nFolders, subplot_index, gap, marg_h, marg_w);
        imshow(img, []);
        axis image off;
        
        % Draw a colored boundary on the diagonal.
        % Here, the diagonal means: for each folder, if focal index equals the same number as the folder index.
        % (This is just an example; adjust the condition if desired.)
        if row == col
            hold on;
            rectangle('Position', [0.5, 0.5, size(img,2), size(img,1)], ...
                      'EdgeColor', colors{col}, 'LineWidth', 4);
            hold off;
        end
    end
end

%% Hyperspectral simulation - 3DOptiX
baseFolder = '/home/samiarja/Desktop/PhD/Code/OctoEye/simulation/';
folderInfo = dir(baseFolder);
isSubFolder = [folderInfo.isdir] & ~ismember({folderInfo.name}, {'.', '..'});
focalFolders = folderInfo(isSubFolder);

wavelengthData = containers.Map('KeyType','char','ValueType','any');

for i = 1:numel(focalFolders)
    folderName = focalFolders(i).name;
    focalPath = fullfile(baseFolder, folderName);
    focal_mm = str2double(folderName);
    
    % List all CSV files in the folder
    csvFiles = dir(fullfile(focalPath, '*.csv'));
    
    for j = 1:numel(csvFiles)
        csvFileName = csvFiles(j).name;
        fullFilePath = fullfile(focalPath, csvFileName);
        
        % Read the CSV file, skipping the first 5 header rows.
        % (Assumes that the numerical image data starts at row 6.)
        try
            data = readmatrix(fullFilePath, 'NumHeaderLines', 5);
        catch
            % Fallback using csvread if necessary (for older MATLAB versions)
            data = csvread(fullFilePath, 5, 0);
        end
        
        % Skip if data is empty
        if isempty(data)
            continue;
        end
        
        % Compute the maximum pixel intensity in the data
        maxIntensity = max(data(:));
        
        % Extract the wavelength from the file name (remove the '.csv' extension)
        [~, wavelengthStr, ~] = fileparts(csvFileName);
        
        % If this wavelength key already exists, append the new data; otherwise create it.
        if wavelengthData.isKey(wavelengthStr)
            currentData = wavelengthData(wavelengthStr);
            currentData = [currentData; focal_mm, maxIntensity];
            wavelengthData(wavelengthStr) = currentData;
        else
            wavelengthData(wavelengthStr) = [focal_mm, maxIntensity];
        end
    end
end

%% Prepare and plot the data
% Define the base folder containing all focal-length folders
baseFolder = '/home/samiarja/Desktop/PhD/Code/OctoEye/simulation/';

% Define the wavelength files and corresponding colors (color palette)
files = { '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000' };
colors = { '#610061', 'b', '#00ff92', 'g', '#ffbe00', 'r', '#e90000', '#a10000', '#6d0000', '#3b0f0f', '#210808', '#1c0404', '#030000' };

% Get list of subdirectories (each focal length folder)
folderInfo = dir(baseFolder);
isSubFolder = [folderInfo.isdir] & ~ismember({folderInfo.name}, {'.', '..'});
focalFolders = folderInfo(isSubFolder);

% Initialize a containers.Map to store data for each wavelength.
% Each key is a wavelength (string, e.g., '400') and the value is an array of rows: [focal_length_mm, max_intensity]
wavelengthData = containers.Map('KeyType','char','ValueType','any');

% Loop over each focal length folder
for i = 1:numel(focalFolders)
    folderName = focalFolders(i).name;
    focalPath = fullfile(baseFolder, folderName);
    focal_mm = str2double(folderName);
    
    % List all CSV files in the current folder
    csvFiles = dir(fullfile(focalPath, '*.csv'));
    
    for j = 1:numel(csvFiles)
        csvFileName = csvFiles(j).name;
        fullFilePath = fullfile(focalPath, csvFileName);
        
        % Read the CSV file, skipping the first 5 header rows.
        try
            data = readmatrix(fullFilePath, 'NumHeaderLines', 5);
        catch
            data = csvread(fullFilePath, 5, 0);
        end
        
        if isempty(data)
            continue;
        end
        
        % Compute the maximum intensity from the image pixel data
        maxIntensity = max(data(:));
        
        % Extract wavelength from the file name (assumes name like '400.csv')
        [~, wavelengthStr, ~] = fileparts(csvFileName);
        
        % Append the [focal_mm, maxIntensity] to the corresponding wavelength entry
        if wavelengthData.isKey(wavelengthStr)
            currentData = wavelengthData(wavelengthStr);
            currentData = [currentData; focal_mm, maxIntensity];
            wavelengthData(wavelengthStr) = currentData;
        else
            wavelengthData(wavelengthStr) = [focal_mm, maxIntensity];
        end
    end
end

%% Plot hyperspectral response from simulation
% Prepare and plot the data using the specified wavelength order
figure(456457); clf; hold on;
set(gca, 'LineWidth', 2, 'FontSize', 16);
legendHandles = [];
legendLabels = {};

% Loop only over wavelengths defined in "files" (sorted order from 400nm to 1000nm)
for idx = 1:numel(files)
    key = files{idx};
    if wavelengthData.isKey(key)
        dataMat = wavelengthData(key);  % Each row: [focal_mm, maxIntensity]
        
        % Sort the data by focal length (mm)
        dataMat = sortrows(dataMat, 1);
        
        % Convert focal length from mm to cm for plotting
        x = dataMat(:,1);
        y = dataMat(:,2);

        windowSize = max(3, round(0.1 * numel(y)));
        y = smoothdata(y, 'movmean', windowSize);
        
        % Get the corresponding color from the palette
        col = colors{idx};
        
        % Plot a low-opacity shaded area under the curve
        area(x, y, 'FaceColor', col, 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
        
        % Plot the line with markers and store the handle for the legend
        h = plot(x, y, 'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8);
        legendHandles(end+1) = h; %#ok<SAGROW>
        legendLabels{end+1} = key; %#ok<SAGROW>
    end
end

% Reverse the x-axis so that higher focal lengths (cm) appear on the left.
% set(gca, 'XDir', 'reverse');
xlabel('Focal Length (cm)', 'FontSize', 24);
ylabel('Intensity', 'FontSize', 24);
title("Hyperspectral Response - Simulation");
grid on;

% Rotate the x-axis tick labels by 45°.
xtickangle(45);

% Create a legend with one entry per wavelength dataset in sorted order.
legend(legendHandles, legendLabels, 'Location', 'east', 'FontSize', 24);

% % Add text annotations under the x-axis.
% text(0, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'r');
% text(1, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'r');
xlim([59.5 63.5]);
% ylim([5 100]);
hold off;


focal_peak_all = [];
figure(45645234); clf; hold on;
set(gca, 'LineWidth', 2, 'FontSize', 16);
legendHandles = [];
legendLabels = {};
% Loop only over wavelengths defined in "files" (sorted order from 400nm to 1000nm)
for idx = 1:numel(files)
    key = files{idx};
    if wavelengthData.isKey(key)
        dataMat = wavelengthData(key);  % Each row: [focal_mm, maxIntensity]
        
        % Sort the data by focal length (mm)
        dataMat = sortrows(dataMat, 1);
        
        % Convert focal length from mm to cm for plotting
        x = dataMat(:,1);
        y = dataMat(:,2);

        windowSize = max(3, round(0.1 * numel(y)));
        y = smoothdata(y, 'movmean', windowSize);

        [peak_val, peak_idx] = max(y);
        focal_peak = x(peak_idx);

        focal_peak_all = [focal_peak_all; focal_peak];
        
        % Get the corresponding color from the palette
        col = colors{idx};
        
        subtightplot(13,1,idx)
        % Plot a low-opacity shaded area under the curve
        area(x, y, 'FaceColor', col, 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');hold on;
        h = plot(x, y, 'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8);

        legendHandles(end+1) = h; %#ok<SAGROW>
        legendLabels{end+1} = key; %#ok<SAGROW>
        xlim([59.5 63.5]);grid on;hold off;
        legend(h, files{idx}, 'Location', 'west', 'FontSize', 24);
    end
end

% xlabel('Focal Length (cm)', 'FontSize', 24);
% ylabel('Intensity', 'FontSize', 24);
% title("Hyperspectral Response - Simulation");
% xtickangle(45);

% % Create a legend with one entry per wavelength dataset in sorted order.
% legend(legendHandles, legendLabels, 'Location', 'east', 'FontSize', 24);

% Hyperspectral curve - Simulation
addpath("hex2rgb.m")

wavelength = 400:50:1000;
optimal_focal = focal_peak_all;
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

figure(567567);
clf;
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
lineDiameter = 0.05;    % Total thickness (adjust as needed)
numCross = 55;           % Number of cross-section points across the tube

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
xlabel('Focal Length (cm)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
title("Hyperspectral Response - Simulation");

% --- Add rotated text annotations for spectral regions ---
text(60.82, 700, 'Visible light', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);
text(60.82, 950, 'Infrared', 'FontSize', 24, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'k', 'Rotation', 90);

text(0, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 16, 'Color', 'r');
text(1, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 16, 'Color', 'r');

grid on;
hold off;
xlim([60.72 61.815]);
ylim([400 1000])

%% 13x13 grid simulation data

% Intensity and ROI parameters
pixel_intensity_limit = 1300;
ROI_size = 10;         % Size of the region used for the initial ROI (square)
ROI_half = ROI_size / 2;

% Subplot layout parameters (using subtightplot)
gap    = [0.005 0.005];  % [vertical_gap, horizontal_gap]
marg_h = [0.02 0.02];    % [bottom_margin, top_margin]
marg_w = [0.02 0.02];    % [left_margin, right_margin]

% Base folder with simulation CSV files (each subfolder is a focal-length value)
baseFolder = '/home/samiarja/Desktop/PhD/Code/OctoEye/simulation/';
load(baseFolder+"focal_peak_all.mat");

% Define the wavelength filenames and corresponding colors
files = { '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000' };
colors = { '#610061', 'b', '#00ff92', 'g', '#ffbe00', 'r', '#e90000', '#a10000', '#6d0000', '#3b0f0f', '#210808', '#1c0404', '#030000' };

% Provided focal peak (one per wavelength; units must match the folder names)

nWavelengths = numel(files);  % should be 13

% DETERMINE FOCAL FOLDER NAMES AND VALUES

% List all subdirectories in baseFolder (each should be named as a focal length)
folderInfo = dir(baseFolder);
isSubFolder = [folderInfo.isdir] & ~ismember({folderInfo.name}, {'.','..'});
focalFolders = folderInfo(isSubFolder);

% Extract numeric focal values from folder names
focalValues = [];
validFolderNames = {};
for k = 1:numel(focalFolders)
    val = str2double(focalFolders(k).name);
    if ~isnan(val)
        focalValues(end+1) = val;  %#ok<SAGROW>
        validFolderNames{end+1} = focalFolders(k).name;  %#ok<SAGROW>
    end
end

% Sort the focal values (and corresponding folder names)
[focalValuesSorted, sortIdx] = sort(focalValues);
folderNames = validFolderNames(sortIdx);

% FIND THE FOLDER (FOCAL SETTING) CLOSEST TO EACH PEAK

% For each wavelength (from files), find the folder whose focal value is closest
% to the provided focal_peak_all value.
peakFolderIndices = zeros(nWavelengths, 1);
for i = 1:nWavelengths
    [~, idx] = min(abs(focalValuesSorted - focal_peak_all(i)));
    peakFolderIndices(i) = idx;
end

figure(1001); clf;
set(gcf, 'Color', 'w');

for row = 1:nWavelengths
    % Base folder for this row (closest to the peak for wavelength files{row})
    baseFolderName = folderNames{ peakFolderIndices(row) };
    for col = 1:nWavelengths
        currWavelength = files{col};
        % Build full path to CSV file in the selected folder
        csvPath = fullfile(baseFolder, baseFolderName, [currWavelength, '.csv']);
        
        % Read the CSV file (skip the first 5 header rows)
        try
            imgData = readmatrix(csvPath, 'NumHeaderLines', 5);
        catch
            imgData = csvread(csvPath, 5, 0);
        end
        
        if isempty(imgData)
            continue;
        end
                
        % Get image size and center
        [r_img, c_img] = size(imgData);
        center_y = round(r_img/2);
        center_x = round(c_img/2);
        
        % Define ROI around the image center
        roi_y_min = max(1, center_y - ROI_half + 1);
        roi_y_max = min(r_img, center_y + ROI_half);
        roi_x_min = max(1, center_x - ROI_half + 1);
        roi_x_max = min(c_img, center_x + ROI_half);
        ROI_img = imgData(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
        
        % Smooth the ROI with a Gaussian filter to reduce spikes
        final_ROI = ROI_img; %imgaussfilt(ROI_img, 3);        
        
        % Determine subplot index (row-major order)
        subplot_index = (row - 1) * nWavelengths + col;
        % Use subtightplot if available; otherwise, use subplot
        if exist('subtightplot','file')
            subtightplot(nWavelengths, nWavelengths, subplot_index, gap, marg_h, marg_w);
        else
            subplot(nWavelengths, nWavelengths, subplot_index);
        end
        
        % Display the (optionally further smoothed) ROI image.

        % imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
        imshow(final_ROI, [0 pixel_intensity_limit]);
        
        % If this is a diagonal cell, add a thick colored boundary.
        if row == col
            hold on;
            rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                      'EdgeColor', colors{col}, 'LineWidth', 4);
            hold off;
        end
    end
end


%% Temporal derivative
wavelengths = 600:50:1000;
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
figure(45456);clf();
for i = 1:numel(wavelengths)
    fullfilname = "/media/samiarja/USB/Optical_characterisation/frame_to_events_simulation/"+num2str(wavelengths(i))+"/"+num2str(wavelengths(i))+"_event_rate_data.mat";
    if exist(fullfilname, 'file')
        load(fullfilname)
        windowSize = max(3, round(0.1 * numel(samples)));
        samples = smoothdata(samples, 'movmean', windowSize);

        plot(timestamps,samples,'Color',colors{i},"LineWidth",2);hold on;grid on
    end

end

%% Convert 3DOptiX .csv to .png frames (NO NEED TO RUN AGAIN)
baseFolder = '/home/samiarja/Desktop/PhD/Code/OctoEye/simulation/';
folderInfo = dir(baseFolder);
isSubFolder = [folderInfo.isdir] & ~ismember({folderInfo.name}, {'.', '..'});
focalFolders = folderInfo(isSubFolder);

% Extract numeric focal lengths from folder names.
numFocal = numel(focalFolders);
focalLengths = zeros(numFocal, 1);
for i = 1:numFocal
    focalLengths(i) = str2double(focalFolders(i).name);
end

% Filter to include only focal lengths between 59.505 and 63.615.
filterIdx = focalLengths >= 59.505 & focalLengths <= 63.615;
focalFolders = focalFolders(filterIdx);
focalLengths = focalLengths(filterIdx);

% Sort focal folders numerically (ensuring three-decimal ordering).
[~, sortIdx] = sort(focalLengths);
focalFolders = focalFolders(sortIdx);
focalLengths = focalLengths(sortIdx);

% Define the base output folder.
output_folder = "/media/samiarja/USB/Optical_characterisation/3DOptiX_simulation/";

% Loop over wavelengths from 400 to 1000 in increments of 50.
for wavelength = 400:50:1000
    wavelengthStr = num2str(wavelength);
    outputWavelengthFolder = fullfile(output_folder, wavelengthStr);
    if ~exist(outputWavelengthFolder, 'dir')
        mkdir(outputWavelengthFolder);
    end
    
    % First Pass: Determine Global Maximum Intensity for This Wavelength
    globalMax = -inf;
    for i = 1:numel(focalFolders)
        folderName = focalFolders(i).name;
        focalPath = fullfile(baseFolder, folderName);
        csvFilePath = fullfile(focalPath, [wavelengthStr, '.csv']);
        if ~exist(csvFilePath, 'file')
            continue;
        end
        try
            data = readmatrix(csvFilePath, 'NumHeaderLines', 5);
        catch
            data = csvread(csvFilePath, 5, 0);
        end
        if isempty(data)
            continue;
        end
        currentMax = max(data(:));
        if currentMax > globalMax
            globalMax = currentMax;
        end
    end
    
    if isinf(globalMax)
        fprintf('No data found for wavelength %s.\n', wavelengthStr);
        continue;
    end
    
    % Second Pass: Process Each Focal Folder, Scale Using Global Maximum, and Save
    for i = 1:numel(focalFolders)
        folderName = focalFolders(i).name;
        focalPath = fullfile(baseFolder, folderName);
        csvFilePath = fullfile(focalPath, [wavelengthStr, '.csv']);
        
        if ~exist(csvFilePath, 'file')
            fprintf('File %s not found in folder %s. Skipping.\n', [wavelengthStr, '.csv'], folderName);
            continue;
        end
        
        try
            data = readmatrix(csvFilePath, 'NumHeaderLines', 5);
        catch
            data = csvread(csvFilePath, 5, 0);
        end
        
        if isempty(data)
            fprintf('No data found in %s. Skipping.\n', csvFilePath);
            continue;
        end
        
        % Clip (if necessary) and scale the data so that the global maximum maps to 255.
        % (Here we use min(data, globalMax) in case some values are above the global maximum.)
        dataClipped = min(data, globalMax);
        img = uint8(255 * double(dataClipped) / double(globalMax));
        
        % Format the focal length to three decimals for the filename.
        focal_mm = str2double(folderName);
        outputImageName = sprintf('%.3f.png', focal_mm);
        outputImageFile = fullfile(outputWavelengthFolder, outputImageName);
        
        imwrite(img, outputImageFile);
        fprintf('Saved image %s for wavelength %s from folder %s\n', outputImageName, wavelengthStr, folderName);
    end
end

%% Analyse the 3DOptiX event rate data
% Define parent directory and color palette for each wavelength
parent_dir = "/media/samiarja/USB/Optical_characterisation/3DOptiX_simulation_events/";
colors   = {
    '#610061', ... % 400nm
    'b', ...      % 450nm
    '#00ff92', ...% 500nm
    'g', ...      % 550nm
    '#ffbe00', ...% 600nm
    'r', ...      % 650nm
    '#e90000', ...% 700nm
    '#a10000', ...% 750nm
    '#6d0000', ...% 800nm
    '#3b0f0f', ...% 850nm
    '#210808', ...% 900nm
    '#1c0404', ...% 950nm
    '#030000'     % 1000nm
};

% Create figure with white background and clear previous content
figure(66); clf;
set(gcf, 'Color', 'w');

hold on;
legendLabels = cell(1, numel(400:50:1000));
c = 1;

% Loop through wavelengths and plot each event rate vs. time curve
for wavelength = 400:50:1000
    % Construct the file path and load the data variables 'timestamps' and 'samples'
    dataPath = fullfile(parent_dir, num2str(wavelength), [num2str(wavelength) '_event_rate_data.mat']);
    load(dataPath, 'timestamps', 'samples');
    
    y = samples;
    windowSize = max(1, round(0.1 * numel(y)));  % Calculate smoothing window size
    y = smoothdata(y, 'movmean', windowSize);     % Apply moving average smoothing
    
    % Plot the data with the designated color and increased line width for clarity
    p = plot(timestamps, y, "Color", colors{c}, "LineWidth", 2);
    % Optionally, add markers to emphasize data points:
    % p.Marker = 'o'; p.MarkerSize = 4; p.MarkerFaceColor = colors{c};
    
    legendLabels{c} = sprintf('%dnm', wavelength); % Build legend label
    c = c + 1;
end

% Set axes properties for a refined appearance
set(gca, 'YScale', 'log', 'FontSize', 14, 'LineWidth', 1.5, 'TickDir', 'out', 'TickLength', [0.02 0.02]);
xlim([0 4.5e5]);
grid on; grid minor;
box on;

% Add descriptive labels and title with bold and larger fonts
xlabel('Time (\mus)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Event Rate (Log scale)', 'FontSize', 16, 'FontWeight', 'bold');
% title('Event Rate vs. Time for Various Wavelengths', 'FontSize', 18, 'FontWeight', 'bold');

% Add a legend in the best location
legend(legendLabels, 'Location', 'best');

%% Simulation 3D plot vs WAVELENGTHS (Hourglass shape)
addpath("myColorToRGB")
% Define colors for each wavelength
colors = { ...
    '#610061', ... % 400nm
    'b', ...       % 450nm
    '#00ff92', ...% 500nm
    'g', ...       % 550nm
    '#ffbe00', ...% 600nm
    'r', ...       % 650nm
    '#e90000', ...% 700nm
    '#a10000', ...% 750nm
    '#6d0000', ...% 800nm
    '#3b0f0f', ...% 850nm
    '#210808', ...% 900nm
    '#1c0404', ...% 950nm
    '#030000'      % 1000nm
};

% Create a figure with a white background
figure(4554); clf;
set(gcf, 'Color', 'w');  % White figure background

% Configure axes for a white theme with black labels
ax = gca;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', ...
    'FontSize', 14, 'LineWidth', 1.5);
grid on;
hold on;

% Loop over each wavelength and load the corresponding events data
c = 1;
wavelengths = 400:50:1000;
for wavelength = wavelengths
    % Construct file path and load events data (assumes variable "events" exists)
    dataPath = fullfile("/media/samiarja/USB/Optical_characterisation/3DOptiX_simulation_events/", ...
        num2str(wavelength), [num2str(wavelength) '_ev_100_10_100_40_0.1_0.01_tiff_dvs_without_hot_pixels_crop.mat']);
    load(dataPath, 'events');
    
    % Convert the color string to an RGB triple
    colRGB = myColorToRGB(colors{c});
    
    % Plot the events using scatter3.
    % x coordinate: events(:,2)
    % time coordinate: events(:,1)
    % y coordinate: events(:,3)
    h = scatter3(events(:,2), events(:,1), events(:,3), 20, colRGB, 'filled');
    
    % Set marker transparency for an artistic look
    h.MarkerFaceAlpha = 0.7;
    
    c = c + 1;
end
xlim([10 25]);

% Label axes with black text
xlabel("x [px]", 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
ylabel("time [\mu s]", 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
zlabel("y [px]", 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');

% Set the view and camera projection for a clear 3D perspective
view([80 20]);
camproj perspective;
camorbit(20, 10, 'data', [0 1 0]);

% Add a legend mapping wavelengths to colors (labels in black)
legendStrings = arrayfun(@(w) sprintf('%dnm', w), wavelengths, 'UniformOutput', false);
legend(legendStrings, 'TextColor', 'k', 'Location', 'best');

%% Real event data 3D plot vs WAVELENGTHS (Hourglass shape)
wavelength = 600;

load("/media/samiarja/USB/gen4-windows/recordings/"+num2str(wavelength)+"/2025-02-21T10-49-52Z_dvs_without_hot_pixels_crop.mat")

figure(6565);
plot3(events(:,2),events(:,3),events(:,1),".b","MarkerSize",5);
xlabel("x");ylabel("y");zlabel("t");zlim([0 1e8]);grid on
% xlim([200 400]);ylim([150 300]);





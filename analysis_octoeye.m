%% Figure 1: RGB distribution
% quantum_efficiency_cam = [0.73, 0.71, 0.45];
pixel_intensity_limit = 1100;
quantum_efficiency_cam = [1,1,1];
blue_qe  = quantum_efficiency_cam(1);
green_qe = quantum_efficiency_cam(2);
red_qe   = quantum_efficiency_cam(3);

parent_path = 'D:\Optical_characterisation\Hyperspectral_new\';

% Read images
red_focal_length_red_wavelength    = imread(parent_path + "650\image_30.tiff");
red_focal_length_green_wavelength  = imread(parent_path + "550\image_30.tiff");
red_focal_length_blue_wavelength   = imread(parent_path + "450\image_30.tiff");

green_focal_length_red_wavelength   = imread(parent_path + "650\image_33.tiff");
green_focal_length_green_wavelength = imread(parent_path + "550\image_33.tiff");
green_focal_length_blue_wavelength  = imread(parent_path + "450\image_33.tiff");

blue_focal_length_red_wavelength    = imread(parent_path + "650\image_37.tiff");
blue_focal_length_green_wavelength  = imread(parent_path + "550\image_37.tiff");
blue_focal_length_blue_wavelength   = imread(parent_path + "450\image_37.tiff");

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
shift = 50;  % (adjust as needed)
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
ylabel('Intensity', 'FontSize', 18);
grid on;

% Only include the stored line handles in the legend (one per wavelength)
legend([h_red, h_green, h_blue], {'650nm wavelength', '550nm wavelength', '450nm wavelength'}, 'Location', 'Best');

hold off;ylim([100 950]);


%% Greylevel hyperspectral analysis
feedback_range_array = 2600:1:3000;
parent_path = 'D:\Optical_characterisation\hyperspectral_new\';
folders = {
    '400',...
    % '425',...
    '450',...
    % '475', ...
    '500', ...
    % '525',...
    '550',...
    % '575',...
    '600',...
    % '625', ...
    '650',...
    % '675',...
    '700',...
    % '725',...
    '750',...
    % '775',...
    '800',...
    % '825',...
    '850',...
    % '875',...
    '900',...
    % '925',...
    '950',...
    % '975',...
    '1000'
    };      % Folder names (wavelengths)

colors   = {
            '#610061', ... %400nm
            % '#5400ff',... %425nm
            'b',...       % 450nm
            % 'c',...       % 475nm
            '#00ff92',... %500nm
            % '#4aff00', ... %525nm
            'g',...       %550nm
            % '#f0ff00', ... % 575nm
            '#ffbe00', ... %600nm
            % '#ff6300', ... 625nm
            'r',...       % 650nm
            % '#ff0000', ... %675nm
            '#e90000', ... %700nm
            % '#d10000', ... %725nm
            '#a10000',...%750nm
            % '#a10000',...%775nm
            '#6d0000',...%800nm
            % '#4f1515',...%825nm
            '#3b0f0f',...%850nm
            % '#2b0b0b',...%875nm
            '#210808',...%900nm
            % '#1c0606',...%925nm
            '#1c0404',...%950nm
            % '#140202',...%975nm
            '#030000',%1000nm
            };             

% Quantum efficiency factors for each folder:
% For folder "450": use blue QE, "550": green QE, "650": red QE.
quantum_efficiency_cam = [ 
                            1, ... %400nm
                            % 1, ... %425nm
                            1, ... %450nm
                            % 1,... %475nm
                            1, ... %500nm
                            % 1, ... %525nm
                            1, ... %550nm
                            % 1, ... % 575nm
                            1, ... %600nm
                            % 1,... %625nm 
                            1, ... %650nm
                            % 1, ... %675nm
                            1, ... %700nm
                            % 1,...%725nm
                            1,...%750nm
                            % 1,...%775nm
                            1,...%800nm
                            % 1,...%825nm
                            1,...%850nm
                            % 1,...%875nm
                            1,...%900nm
                            % 1,...%925nm
                            1,...%950nm
                            % 1,...%975nm
                            1,...%1000nm
                            ]; % BCGR

QE = quantum_efficiency_cam;

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

        img(img > 60000) = 0;
        
        % Crop a vertical region around the image center (±30 rows)
        [rows, ~] = size(img);
        mid_row = round(rows/2);
        r_start = max(1, mid_row - 300);
        r_end   = min(rows, mid_row + 300);
        % cropped = double(img(r_start:r_end, :));

        cropped = double(img);

        % For each row in the cropped region, find its maximum pixel value.
        rowMaxima = max(cropped, [], 2);
        peak = max(rowMaxima);

        % figure(45654);imagesc(cropped);colorbar;title(fileName)
        
        % Apply the appropriate quantum efficiency correction.
        % Folder "450" uses blue (QE(1)), "550" uses green (QE(2)), "650" uses red (QE(3)).
        peak_corrected = peak / QE(i);
        
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

% Rotate the x-axis tick labels by 45°.
xtickangle(45);

% Create a legend with only three entries (one per dataset).
legend(h, { ...
    '400nm', ...
    % '425nm', ...
    '450nm', ...
    % '475nm', ...
    '500nm', ...
    % '525nm', ...
    '550nm', ...
    % '575nm', ...
    '600nm', ...
    % '625nm', ...
    '650nm', ...
    % '675nm', ...
    '700nm', ...
    % '725nm', ...
    '750nm', ...
    % '775nm', ...
    '800nm', ...
    % '825nm', ...
    '850nm', ...
    % '875nm', ...
    '900nm', ...
    % '925nm', ...
    '950nm', ...
    % '975nm', ...
    '1000nm', ...
    }, 'Location', 'west', 'FontSize', 24);

% Add tags under the x-axis.
text(0, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'k');
text(1, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'k');

hold off;xlim([3.28 3.365]);ylim([450 1100]);

%% 13x13 grid for each wavelength and focal length
% Greylevel hyperspectral analysis
feedback_range_array = 2600:10:3400;
pixel_intensity_limit = 1200; %65535/2;

parent_path = 'D:\Optical_characterisation\Hyperspectral_new\';
% Use 12 folders (wavelengths)
folders = {
    '400',...
    % '425',...
    '450',...
    % '475', ...
    '500', ...
    % '525',...
    '550',...
    % '575',...
    '600',...
    % '625', ...
    '650',...
    % '675',...
    '700',...
    % '725',...
    '750',...
    % '775',...
    '800',...
    % '825',...
    '850',...
    % '875',...
    '900',...
    % '925',...
    '950',...
    % '975',...
    '1000'
    };      % Folder names (wavelengths)

colors   = {
            '#610061', ... %400nm
            % '#5400ff',... %425nm
            'b',...       % 450nm
            % 'c',...       % 475nm
            '#00ff92',... %500nm
            % '#4aff00', ... %525nm
            'g',...       %550nm
            % '#f0ff00', ... % 575nm
            '#ffbe00', ... %600nm
            % '#ff6300', ... 625nm
            'r',...       % 650nm
            % '#ff0000', ... %675nm
            '#e90000', ... %700nm
            % '#d10000', ... %725nm
            '#a10000',...%750nm
            % '#a10000',...%775nm
            '#6d0000',...%800nm
            % '#4f1515',...%825nm
            '#3b0f0f',...%850nm
            % '#2b0b0b',...%875nm
            '#210808',...%900nm
            % '#1c0606',...%925nm
            '#1c0404',...%950nm
            % '#140202',...%975nm
            '#030000',%1000nm
            };             

% Quantum efficiency factors for each folder:
% For folder "450": use blue QE, "550": green QE, "650": red QE.
QE_all = [ 
                            1, ... %400nm
                            % 1, ... %425nm
                            1, ... %450nm
                            % 1,... %475nm
                            1, ... %500nm
                            % 1, ... %525nm
                            1, ... %550nm
                            % 1, ... % 575nm
                            1, ... %600nm
                            % 1,... %625nm 
                            1, ... %650nm
                            % 1, ... %675nm
                            1, ... %700nm
                            % 1,...%725nm
                            1,...%750nm
                            % 1,...%775nm
                            1,...%800nm
                            % 1,...%825nm
                            1,...%850nm
                            % 1,...%875nm
                            1,...%900nm
                            % 1,...%925nm
                            1,...%950nm
                            % 1,...%975nm
                            1,...%1000nm
                            ]; % BCGR

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
for i = 1:nFolders
    currFolder = folders{i};
    fileList = dir(fullfile(parent_path, currFolder, '*.tiff'));
    fileListAll{i} = fileList;
    nFiles = numel(fileList);
    
    % Preallocate arrays for the current folder.

    % feedback_value = feedback_range_array(j);

    focal_vals = zeros(nFiles,1);
    peak_vals  = zeros(nFiles,1);
    
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
        % Read the image
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        
        img(img > pixel_intensity_limit) = 0;

        [rows, cols] = size(img);
        mid_row = round(rows);
        r_start = max(1, mid_row - vertCrop_half);
        r_end   = min(rows, mid_row + vertCrop_half);
        % cropped = double(img(r_start:r_end, :));
        cropped = double(img);

        rowMaxima = max(cropped, [], 2);
        peak = max(rowMaxima);
        
        % Apply QE correction.
        peak_vals(j) = peak / QE_all(i);
        
    end
    
    % Find the file index with the maximum corrected peak.
    [~, idx_max] = max(peak_vals);
    maxFileIndex{i} = idx_max;
    
    % (Storing focal length data omitted for brevity)
    % focal_mm_all{i} = [];  % not used below
    peak_all{i}     = peak_vals;
    
    fprintf('Folder %s: maximum peak from file index %d (%s)\n', currFolder, idx_max, fileList(idx_max).name);
    feedback_motor_value = [feedback_motor_value;feedback_range_array(idx_max)];
end

% Parameters for the ROI (circle) extraction in each image:
ROI_size = 100;        % Use a 30x30 region (centered on the image center) to search for the brightest pixel.
ROI_half = ROI_size / 2;
crop_radius = 15;     % Final crop: (2*crop_radius+1) x (2*crop_radius+1) region around the detected peak.

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
        img = double(img) / QE_all(col);
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
        imshow(imgaussfilt(final_ROI, 1), [0 1100]);
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
feedback_range_array = 2600:10:3400;
parent_path = 'D:\Optical_characterisation\Hyperspectral_new\';
folders = {
    '400',...
    % '425',...
    '450',...
    % '475', ...
    '500', ...
    % '525',...
    '550',...
    % '575',...
    '600',...
    % '625', ...
    '650',...
    % '675',...
    '700',...
    % '725',...
    '750',...
    % '775',...
    '800',...
    % '825',...
    '850',...
    % '875',...
    '900',...
    % '925',...
    '950',...
    % '975',...
    '1000'
    };      % Folder names (wavelengths)

colors   = {
            '#610061', ... %400nm
            % '#5400ff',... %425nm
            'b',...       % 450nm
            % 'c',...       % 475nm
            '#00ff92',... %500nm
            % '#4aff00', ... %525nm
            'g',...       %550nm
            % '#f0ff00', ... % 575nm
            '#ffbe00', ... %600nm
            % '#ff6300', ... 625nm
            'r',...       % 650nm
            % '#ff0000', ... %675nm
            '#e90000', ... %700nm
            % '#d10000', ... %725nm
            '#a10000',...%750nm
            % '#a10000',...%775nm
            '#6d0000',...%800nm
            % '#4f1515',...%825nm
            '#3b0f0f',...%850nm
            % '#2b0b0b',...%875nm
            '#210808',...%900nm
            % '#1c0606',...%925nm
            '#1c0404',...%950nm
            % '#140202',...%975nm
            '#030000',%1000nm
            };             

% Quantum efficiency factors for each folder:
% For folder "450": use blue QE, "550": green QE, "650": red QE.
quantum_efficiency_cam = [ 
                            0.95, ... %400nm
                            % 1, ... %425nm
                            1, ... %450nm
                            % 1,... %475nm
                            1, ... %500nm
                            % 1, ... %525nm
                            1, ... %550nm
                            % 1, ... % 575nm
                            1, ... %600nm
                            % 1,... %625nm 
                            1.1, ... %650nm
                            % 1, ... %675nm
                            1, ... %700nm
                            % 1,...%725nm
                            1,...%750nm
                            % 1,...%775nm
                            1,...%800nm
                            % 1,...%825nm
                            1,...%850nm
                            % 1,...%875nm
                            1,...%900nm
                            % 1,...%925nm
                            1,...%950nm
                            % 1,...%975nm
                            1,...%1000nm
                            ]; % BCGR

QE = quantum_efficiency_cam;

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
        [rowsImg, ~] = size(img);
        mid_row = round(rowsImg/2);
        r_start = max(1, mid_row - 30);
        r_end   = min(rowsImg, mid_row + 30);
        % cropped = double(img(r_start:r_end, :));

        cropped = double(img);

        
        % For each row in the cropped region, find its maximum pixel value.
        rowMaxima = max(cropped, [], 2);
        peak = max(rowMaxima);
        
        % Apply the QE correction.
        peak_corrected = peak / QE(i);
        
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
edge_fraction_left  = 0.47;
edge_fraction_right = 0.32;

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
% title('Composite Envelope with Horizontal Color Gradient Fill', 'FontSize', 28);

% Add tags under the x-axis.
text(0, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'k');
text(1, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'k');

% grid on;
hold off;
xlim([1.8 3.3]);
ylim([550 1150]);


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
pixel_intensity_limit = 1100;
quantum_efficiency_cam = [1,1,1];
blue_qe  = quantum_efficiency_cam(1);
green_qe = quantum_efficiency_cam(2);
red_qe   = quantum_efficiency_cam(3);

smooth_factor = 5;

% center_x = 1654; center_y = 383; %P1
% center_x = 1699; center_y = 523; %P2
% center_x = 1311; center_y = 411; %P3
% center_x = 958; center_y = 209;  %P4
% center_x = 1393; center_y = 195; %P5
% center_x = 1763; center_y = 177; %P6
% center_x = 1800; center_y = 65;  %P7
% center_x = 1500; center_y = 80;   %P8
center_x = 1000; center_y = 95;   %P9


parent_path = 'D:\Optical_characterisation\longitudinal_chromatic_aberration\9\';


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
    imshow(imgaussfilt(final_ROI, 1), [0 pixel_intensity_limit]);
    hold on;
    rectangle('Position',[0.5, 0.5, size(final_ROI,2), size(final_ROI,1)], ...
                          'EdgeColor', 'r', 'LineWidth', 4);
    hold off;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intensity Distribution Plot
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


%% More precise VIS-IR characterisation
feedback_range_array = 2800:1:3250;
pixel_intensity_limit = 65535;
parent_path = 'D:\Optical_characterisation\hyperspectral_high_resolution\';
folders = {
    '400',...
    % '450',...
    % '500', ...
    % '550',...
    '600',...
    % '650',...
    % '700',...
    % '750',...
    '800',...
    % '850',...
    % '900',...
    % '950',...
    '1000'
    };

colors   = {
            '#610061', ... %400nm
            % 'b',...       % 450nm
            % '#00ff92',... %500nm
            % 'g',...       %550nm
            '#ffbe00', ... %600nm
            % 'r',...       % 650nm
            % '#e90000', ... %700nm
            % '#a10000',...%750nm
            '#6d0000',...%800nm
            % '#3b0f0f',...%850nm
            % '#210808',...%900nm
            % '#1c0404',...%950nm
            '#030000',%1000nm
            };             

% Quantum efficiency factors for each folder:
% For folder "450": use blue QE, "550": green QE, "650": red QE.
quantum_efficiency_cam = [ 
                            1, ... %400nm
                            1, ... %450nm
                            1, ... %500nm
                            1, ... %550nm
                            1, ... %600nm
                            1, ... %650nm
                            1, ... %700nm
                            1,...%750nm
                            1,...%800nm
                            1,...%850nm
                            1,...%900nm
                            1,...%950nm
                            1,...%1000nm
                            ];

QE = quantum_efficiency_cam;

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
        [~, name] = fileparts(fileName);
    

        % feedback_value = str2double(name);
        feedback_value = feedback_range_array(j);

        % Convert raw feedback to focal length (mm) using the dynamic formula:
        %   focal_length_mm = (feedback_max - feedback_value) / (feedback_max - feedback_min) * distance_at_min_mm
        focal_length_mm = feedback_value; % (feedback_max - feedback_value) / feedback_range * distance_at_min_mm;
        
        % Read the image.
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end

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


        % figure(45654);imagesc(cropped);colorbar;title(fileName)

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
    x = focal_mm_all{i} / 1;  % x-axis: focal length in cm
    y = peak_all{i};           % y-axis: corrected peak intensity
    col = colors{i};
    
    % Plot a low-opacity shaded area under the curve.
    area(x, y, 'FaceColor', col, 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'HandleVisibility','off');
    
    % Overlay the line with markers.
    h(i) = plot(x, y,'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8);
end

% Reverse the x-axis so that higher focal lengths (cm) appear on the left.
% set(gca, 'XDir', 'reverse');

xlabel('Focal Length (cm)', 'FontSize', 24);
ylabel('Intensity', 'FontSize', 24);
grid on;

% Rotate the x-axis tick labels by 45°.
xtickangle(45);

% Create a legend with only three entries (one per dataset).
legend(h, folders, 'Location', 'east', 'FontSize', 24);

% Add tags under the x-axis.
text(0, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 24, 'Color', 'k');
text(1, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 24, 'Color', 'k');


hold off;
% ylim([7 35]);
% xlim([feedback_range_array(1) 3250]);
% xlim([2.3 3.3]);
% ylim([0 80]);

%% test script for hyres hyperspectral
parent_path = 'D:\Optical_characterisation\hyperspectral_high_resolution\1000';

fileList = dir(fullfile(parent_path, '*.tiff'));
nFiles = numel(fileList);

for j = 1:nFiles
    fileName = fileList(j).name;
    [~, name] = fileparts(fileName);

    img = imread(fullfile(parent_path, fileName));
    if size(img,3) == 3
        img = rgb2gray(img);
    end

    img(img > 500000) = 0;
        
    % Crop a vertical region around the image center (±30 rows)
    [rows, ~] = size(img);
    mid_row = round(rows/2);
    r_start = max(1, mid_row - 100);
    r_end   = min(rows, mid_row + 50);
    cropped = double(img(r_start:r_end, :));

    cropped = imgaussfilt(cropped, 2);

    % cropped = double(img);

    figure(5678);

    imagesc(cropped);
    title(num2str(j));colorbar;
    % pause(1)
    
end


%% Event based hyperspectral peaks and rainbow plot
data = load('D:\gen4-windows\recordings\event_based_hyperspectral_results.mat');

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
set(gcf, 'Color', 'w', 'Position', [100 100 650 600]);
hold on;

% Create a patch to fill under the curve with opacity.
baseline = 1000;  % Baseline at 1000 nm.
patch_x = optimal_focal;
patch_y = wavelength;
patch_x = [patch_x, fliplr(optimal_focal)];
patch_y = [patch_y, repmat(baseline, 1, numel(optimal_focal))];
norm_wavelength = (wavelength - min(wavelength)) / (max(wavelength) - min(wavelength));
patch_c = [norm_wavelength, ones(1, numel(norm_wavelength))];
patch(patch_x, patch_y, patch_c, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Re-sample the discrete data to create a smooth gradient line.
t_fine = linspace(min(optimal_focal), max(optimal_focal), 1000);
w_fine = interp1(optimal_focal, double(wavelength), t_fine, 'linear'); 
norm_w_fine = (w_fine - min(wavelength)) / (max(wavelength) - min(wavelength));

xx = [t_fine; t_fine];
yy = [w_fine; w_fine];
zz = zeros(size(xx));
cc = [norm_w_fine; norm_w_fine];

surface(xx, yy, zz, cc, 'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 4);
colormap(interp_colormap);

% Overlay markers at each original data point.
for i = 1:length(optimal_focal)
    scatter(optimal_focal(i), wavelength(i), 100, customRGB(i,:), 'filled', 'MarkerEdgeColor', 'k');
end

% Add a horizontal dashed line at 750 nm.
yline(750, '--k', 'LineWidth', 2);

set(gca, 'YDir', 'reverse');  % Lower wavelengths appear at the top.
set(gca, 'YTick', 400:50:1000, 'FontSize', 16, 'LineWidth', 3);
xlabel('Focal Length (cm)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');

% Add rotated text annotations for spectral regions.
text(3.173, 700, 'Visible light', 'FontSize', 18, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'w', 'Rotation', 90);
text(3.173, 875, 'Infrared', 'FontSize', 18, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'w', 'Rotation', 90);

text(0, -0.1, '(Close to ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'FontSize', 12, 'Color', 'r');
text(1, -0.1, '(Far from ball lens)', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'FontSize', 12, 'Color', 'r');

grid on;
hold off;
xlim([3.163 3.4]);

function rgb = hex2rgb(hexStr)
    if hexStr(1) == '#'
        hexStr = hexStr(2:end);
    end
    r = double(hex2dec(hexStr(1:2)))/255;
    g = double(hex2dec(hexStr(3:4)))/255;
    b = double(hex2dec(hexStr(5:6)))/255;
    rgb = [r, g, b];
end


%% 13x13 hyperspectral image grid





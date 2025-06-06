%% 13x13 grid for each wavelength and focal length - simulation
parent_folder = "/media/samiarja/USB/OctoEye_paper_dataset/";
parent_path = parent_folder+'./octopus_matlab_simulator/simulated_frames/';

pixel_intensity_limit = 1;
feedback_range_array = 1:1:200;

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
    };

colors   = {
    '#610061', ... %400nm
    'b',...       %450nm
    '#00ff92',... %500nm
    'g',...       %550nm
    '#ffbe00', ...%600nm
    'r',...       %650nm
    '#e90000', ...%700nm
    '#a10000',... %750nm
    '#6d0000',... %800nm
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

% Loop Over Each Folder to determine the file with the maximum corrected peak.
% Also store the peak value for each folder.
vertCrop_half = 30;  % crop ±30 rows about the image center

feedback_motor_value = [];
% Each row: [wavelength, focal (cm), peak_intensity]
frame_camera_hyperspectral_output = zeros(nFolders,3);
for i = 1:nFolders
    currFolder = folders{i};
    fileList = dir(fullfile(parent_path, currFolder, '*.png'));
    fileListAll{i} = fileList;
    nFiles = numel(fileList);

    focal_vals = zeros(nFiles,1);
    peak_vals  = zeros(nFiles,1);
    
    % Sort fileList based on the numeric portion of the filename.
    numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
    [~, sortIdx] = sort(numbers);
    fileList = fileList(sortIdx);
    
    % Process each file in sorted order.
    for j = 1:nFiles
        fileName = fileList(j).name;
        % Use feedback_range_array(j) as the feedback value.
        feedback_value = feedback_range_array(j);
        
        % Read and convert the image.
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        
        % Crop a vertical region around the image center.
        [rows, ~] = size(img);
        mid_row = round(rows/2);
        r_start = max(1, mid_row - 200);
        r_end   = min(rows, mid_row + 200);
        cropped = double(img(r_start:r_end, :));

        % Blur the cropped region.
        cropped_blurred = imgaussfilt(cropped, 2);

        % For each row, find the maximum pixel value.
        rowMaxima = max(cropped_blurred, [], 2);
        [peak, ~] = max(rowMaxima);
        peak_vals(j) = peak;
    end
    
    % Identify the file with the maximum corrected peak.
    [peak_val, idx_max] = max(peak_vals);
    maxFileIndex{i} = idx_max;
    peak_all{i}     = peak_vals;

    % Compute the focal length using the filename of the max-peak file.
    selected_focal_length = str2double(fileList(idx_max).name(1:4));
    focal_length_mm = (feedback_max - selected_focal_length) / feedback_range * distance_at_min_mm;

    frame_camera_hyperspectral_output(i,:) = [str2double(folders{i}), focal_length_mm/10, peak_val];
    fprintf('Wavelength (nm) %s: Focal length (cm) %d (%s)\n', folders{i}, idx_max, focal_length_mm/10);
    feedback_motor_value = [feedback_motor_value; feedback_range_array(idx_max)];
end

% Compute the global maximum pixel intensity across all folders.
global_max = max(frame_camera_hyperspectral_output(:,3))+50;

% Parameters for ROI (for object extraction):
ROI_size = 100;        % Size of the ROI (centered on the image center) to search for the object.
ROI_half = ROI_size / 2;
crop_radius = 25;      % Final crop: (2*crop_radius+1) x (2*crop_radius+1) region.

% Subplot parameters for subtightplot:
gap    = [0.005 0.005];  % [vertical_gap, horizontal_gap]
marg_h = [0.02 0.02];    % [bottom_margin, top_margin]
marg_w = [0.02 0.02];    % [left_margin, right_margin]

% Create a grid figure using subtightplot.
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
        
        % Sort fileList based on the numeric value in filenames.
        numbers = cellfun(@(nm) str2double(regexp(nm, '\d+', 'match', 'once')), {fileList.name});
        [~, sortIdx] = sort(numbers);
        fileList = fileList(sortIdx);

        fileName = fileList(fileIdx).name;
        img = imread(fullfile(parent_path, currFolder, fileName));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        
        % Convert image to double precision.
        img = double(img);
        [r, c] = size(img);
        center_y = round(r/2);
        center_x = round(c/2);
        
        % Define an ROI about the image center.
        roi_y_min = center_y - ROI_half + 1;
        roi_y_max = center_y + ROI_half;
        roi_x_min = center_x - ROI_half + 1;
        roi_x_max = center_x + ROI_half;
        roi_y_min = max(1, roi_y_min); 
        roi_y_max = min(r, roi_y_max);
        roi_x_min = max(1, roi_x_min); 
        roi_x_max = min(c, roi_x_max);
        ROI_img = img(roi_y_min:roi_y_max, roi_x_min:roi_x_max);
        
        % Optionally smooth the ROI to mitigate hot pixels.
        ROI_smoothed = ROI_img;  % (or use imgaussfilt(ROI_img, 1))
        
        % Compute the weighted centroid of the ROI (the object's center).
        [YY, XX] = ndgrid(1:size(ROI_smoothed,1), 1:size(ROI_smoothed,2));
        total_intensity = sum(ROI_smoothed(:));
        if total_intensity == 0
            % Fallback to the center if the ROI is blank.
            weighted_x_local = (size(ROI_smoothed,2)+1)/2;
            weighted_y_local = (size(ROI_smoothed,1)+1)/2;
        else
            weighted_x_local = sum(XX(:) .* ROI_smoothed(:)) / total_intensity;
            weighted_y_local = sum(YY(:) .* ROI_smoothed(:)) / total_intensity;
        end
        
        % Convert the local ROI centroid to global image coordinates.
        peak_x = roi_x_min - 1 + weighted_x_local;
        peak_y = roi_y_min - 1 + weighted_y_local;
        
        % Define the final crop window centered on the detected centroid.
        desired_size = 2 * crop_radius + 1;
        x_start = round(peak_x) - crop_radius;
        x_end   = round(peak_x) + crop_radius;
        y_start = round(peak_y) - crop_radius;
        y_end   = round(peak_y) + crop_radius;
        
        % Initialize the final ROI with zeros (for padding when needed).
        final_ROI = zeros(desired_size, desired_size);
        
        % Determine the overlapping region between the desired window and the image.
        x_range_orig = max(1, x_start) : min(c, x_end);
        y_range_orig = max(1, y_start) : min(r, y_end);
        
        % Calculate offsets for placing the cropped part into final_ROI.
        x_offset = 1 - min(x_start, 1);
        y_offset = 1 - min(y_start, 1);
        x_range_final = (1 + x_offset):(1 + x_offset + numel(x_range_orig) - 1);
        y_range_final = (1 + y_offset):(1 + y_offset + numel(y_range_orig) - 1);
        
        final_ROI(y_range_final, x_range_final) = img(y_range_orig, x_range_orig);
        
        % Clip the intensities using the global maximum.
        % final_ROI(final_ROI < 0) = 0;
        % final_ROI(final_ROI > global_max) = global_max;
        
        % Compute the subplot index in the grid.
        subplot_index = (row - 1) * nFolders + col;
        subtightplot(nFolders, nFolders, subplot_index, gap, marg_h, marg_w);
        imshow(imgaussfilt(final_ROI, 1), [0 global_max]);
        
        % If the cell lies on the diagonal, add a thick colored boundary.
        if row == col
            hold on;
            rectangle('Position', [0.5, 0.5, desired_size, desired_size], ...
                      'EdgeColor', colors{col}, 'LineWidth', 4);
            hold off;
        end
    end
end
%% event rate analysis - simulator
parent_folder = "/media/samiarja/USB/OctoEye_paper_dataset/";
parent_dir = parent_folder+"./octopus_matlab_simulator/simulated_events/";

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
};

% Create figure with white background and clear previous content
figure(66); clf;
set(gcf, 'Color', 'w');
hold on;

legendLabels = {};
c = 1;

num_groups = 9;
frame_camera_hyperspectral_output = zeros(num_groups, 2);

% peak_all_group = [];
% focal_all_group = [];
% Loop through wavelengths and plot each event rate vs. time curve
for wavelength = 400:50:800
    % Construct the file path for the current wavelength
    dataPath = fullfile(parent_dir, num2str(wavelength), [num2str(wavelength) '_event_rate_data.mat']);
    
    % Check if the file exists
    if exist(dataPath, 'file') == 2
        load(dataPath, 'timestamps', 'samples');
        
        y = samples;
        windowSize = max(1, round(0.1 * numel(y)));  % Calculate smoothing window size
        y = smoothdata(y, 'movmean', windowSize);     % Apply moving average smoothing
        [peakpeak_val peakpeak_idx] = max(y);
        
        frame_camera_hyperspectral_output(c,:)= [wavelength, timestamps(peakpeak_idx)];
        % Plot the data with the designated color and increased line width for clarity
        plot(timestamps, y, "Color", colors{c}, "LineWidth", 4);
        
        % Build legend label and update counter
        legendLabels{end+1} = sprintf('%dnm', wavelength);
        c = c + 1;

        % peak_all_group = [peak_all_group; y];
    else
        fprintf("Data path %s does not exist. Skipping wavelength %dnm.\n", dataPath, wavelength);
    end
end

% focal_all_group = [focal_all_group;timestamps];

% Set axes properties for a refined appearance
set(gca, 'YScale', 'log', 'FontSize', 14, 'LineWidth', 1.5, 'TickDir', 'out', 'TickLength', [0.02 0.02]);
xlim([0 4.5e5]);
grid on; grid minor;
box on;

% Add descriptive labels and title with bold and larger fonts
xlabel('Time (\mus)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Event Rate (Log scale)', 'FontSize', 16, 'FontWeight', 'bold');
title("Per wavelength event rate", 'FontSize', 18, 'FontWeight', 'bold');

% Add a legend if at least one plot was created
if ~isempty(legendLabels)
    h_leg = legend(legendLabels, 'Location', 'best', 'FontSize', 14, 'Box', 'off');
    h_leg.Title.FontSize = 14;
    h_leg.Title.String = 'Wavelengths';

end
xlim([0 1.7e5]);
ylim([2e3 10e6]);


addpath("hex2rgb.m")
wavelength = double(frame_camera_hyperspectral_output(:,1));
optimal_focal = double(frame_camera_hyperspectral_output(:,2));

[sortedFocal, ~] = sort(optimal_focal);
wavelength = double(wavelength);

hex_colors = { '#610061', '#0000FF', '#00ff92', '#00FF00', '#ffbe00', '#FF0000', ...
               '#e90000', '#a10000', '#6d0000'};

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

figure(567534);
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
lineDiameter = 25;    % Total thickness (adjust as needed)
numCross = 500;         % Number of cross-section points across the tube

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

set(gca, 'YDir', 'reverse');  % Lower wavelengths appear at the top
set(gca, 'YTick', 400:50:800, 'FontSize', 16, 'LineWidth', 3);
xlabel({'Distance from sensor to ball lens surface (cm)', '(Focal distance)'}, 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Wavelength (nm)', 'FontSize', 18, 'FontWeight', 'bold');
title("The change in focal distance with wavelength");

grid on;
hold off;
ylim([400 800]);
xlim([1.78e4 3.85e4])
focal_information_event_simulation = [sortedFocal, wavelength]; 

% Parameters and Setup

% Assumed focal simulation information:
% focal_information_event_simulation = [sortedFocal, wavelength];
% (Make sure sortedFocal and wavelength are defined appropriately.)

% Parent directory for the event simulation files:
parent_dir = '/media/samiarja/USB/Optical_characterisation/octopus_matlab_simulator/simulated_events/';

% Folders corresponding to wavelengths (as strings)
folders = {'400', '450', '500', '550', '600', '650', '700', '750', '800'};

% Colors for marking the diagonal cells:
colors = {'#610061', 'b', '#00ff92', 'g', '#ffbe00', 'r', '#e90000', '#a10000', '#6d0000'};

% Time window for event accumulation:
time_window = 1e4;

nFolders = numel(folders);

% First Pass: Accumulate Events & Determine Global Maximum

% Preallocate cell array for event images and a variable for the global maximum.
all_event_images = cell(nFolders, nFolders);
global_max_event = 0;

% Loop over each focal simulation event (rows) and each wavelength (columns).
for row = 1:nFolders
    % Get the selected focal time from the simulation information.
    focal_time = focal_information_event_simulation(row, 1);
    for col = 1:nFolders
        % Get wavelength (from folder name) and build the file path.
        wv = str2double(folders{col});
        dataPath = fullfile(parent_dir, num2str(wv), [num2str(wv) '_ev_100_10_100_40_0_0.01_dvs_without_hot_pixels_crop.mat']);
        load(dataPath, 'events');  % Load variable 'events'
        
        % Filter events based on the focal time and fixed time window.
        filter_idx = events(:,1) >= focal_time & events(:,1) <= focal_time + time_window;
        filtered_events = events(filter_idx, :);
        
        % Accumulate events into an image.
        if isempty(filtered_events)
            event_image = zeros(100, 100);  % Fallback image if no events.
        else
            max_x = max(filtered_events(:,2));
            max_y = max(filtered_events(:,3));
            event_image = zeros(max_x+1, max_y+1);
            for xx = 1:size(filtered_events,1)
                % Add one to indices for MATLAB's 1-based indexing.
                xcoord = filtered_events(xx,2) + 1;
                ycoord = filtered_events(xx,3) + 1;
                event_image(xcoord, ycoord) = event_image(xcoord, ycoord) + 1;
            end
        end
        
        % Store the accumulated image and update the global maximum.
        all_event_images{row, col} = event_image;
        global_max_event = max(global_max_event, max(event_image(:)));
    end
end

% Second Pass: Display the Grid with a Consistent Greyscale Colormap

figure(567567); clf;
set(gcf, 'Color', 'w');

% Subplot layout parameters (using subtightplot)
gap    = [0.005 0.005];  % [vertical_gap, horizontal_gap]
marg_h = [0.02 0.02];    % [bottom_margin, top_margin]
marg_w = [0.02 0.02];    % [left_margin, right_margin]

% Loop over the same grid indices.
for row = 1:nFolders
    for col = 1:nFolders
        event_image = all_event_images{row, col};
        subplot_index = (row - 1) * nFolders + col;
        subtightplot(nFolders, nFolders, subplot_index, gap, marg_h, marg_w);
        
        % Display the event image using the same intensity range for all.
        imagesc(event_image, [0 global_max_event+10]);
        colormap(gray);
        axis off;
        
        % If the subplot lies on the diagonal, add a thick colored border.
        if row == col
            hold on;
            rectangle('Position', [0.5, 0.5, size(event_image,2), size(event_image,1)], ...
                      'EdgeColor', colors{col}, 'LineWidth', 4);
            hold off;
        end
    end
end


parent_folder = "/media/samiarja/USB/OctoEye_paper_dataset/";
data = load(parent_folder+'imx-636-data/event_based_hyperspectral_results.mat'); % rename "recordings" folder to "gen4_data"

% Define the colors for each wavelength.
colors = { ...
    '#610061', ... % 400nm
    'b',       ... % 450nm
    '#00ff92', ... % 500nm
    'g',       ... % 550nm
    '#ffbe00', ... % 600nm
    'r',       ... % 650nm
    '#e90000', ... % 700nm
    '#a10000', ... % 750nm
    '#6d0000', ... % 800nm
    '#3b0f0f', ... % 850nm
    '#210808', ... % 900nm
    '#1c0404', ... % 950nm
    '#030000'      % 1000nm
};

optimal_focal = [3.1633, 3.1818, 3.2445, 3.2927, 3.2957, 3.3198, ...
                 3.3350, 3.3380, 3.3395, 3.3688, 3.3718, 3.3824, 3.3865];

% Choose a scaling factor for the time
time_offset_scale = 1e5;
min_focal = min(optimal_focal);

% Wavelength array for legend labels.
wavelengths = 400:50:1000;
legend_labels = arrayfun(@(w) sprintf('%dnm', w), wavelengths, 'UniformOutput', false);

figure(34545); clf; 
set(gcf, 'Color', 'w');           % Set a white background for the figure.
ax = gca;
hold(ax, 'on');
grid(ax, 'on');
box(ax, 'on');
set(ax, 'FontSize', 16, 'FontName', 'Helvetica', 'LineWidth', 1.5);
view(ax, 3);

% Loop through each event dataset.
for i = 1:13
    % Normalize time so that each event set starts at zero.
    data.events{i}.t = data.events{i}.t - data.events{i}.t(1);
    
    % Compute the time offset based on the optimal focal value.
    time_offset = (optimal_focal(i) - min_focal) * time_offset_scale;
    
    % Bump the time coordinate.
    bumped_t = data.events{i}.t + time_offset;
    
    % Plot the 3D event data with a slightly larger marker for better visibility.
    plot3(data.events{i}.x, data.events{i}.y, bumped_t, ".", ...
          "Color", colors{i}, "MarkerSize", 8, "LineWidth", 1.5);
end

% Enhance the axes with clear labels and a title.
xlabel('x [px]', 'FontSize', 18, 'FontWeight', 'bold', 'FontName', 'Helvetica');
ylabel('y [px]', 'FontSize', 18, 'FontWeight', 'bold', 'FontName', 'Helvetica');
zlabel('Time (\mu s)', 'FontSize', 18, 'FontWeight', 'bold', 'FontName', 'Helvetica');
% title('3D Event Data for Hyperspectral Wavelengths', 'FontSize', 20, 'FontWeight', 'bold', 'FontName', 'Helvetica');

% Add a legend with the wavelengths, placed outside the main plot.
h_leg = legend(legend_labels, 'Location', 'northeastoutside', 'FontSize', 14);
h_leg.Title.String = 'Wavelengths';
% Adjust the axes to fit the data tightly.
axis tight;

% Optionally, adjust the view angle for a better perspective.
camorbit(20, 10);

hold off;

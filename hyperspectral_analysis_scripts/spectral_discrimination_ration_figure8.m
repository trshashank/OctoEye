parent_folder = "/media/samiarja/USB/OctoEye_paper_dataset/";

load(parent_folder+'spectral_descrimination_ratio.mat');
% Removed normalization to display full-scale values.
% R_wavelength = normalize(R_wavelength,"range") ;

% Define the corresponding wavelength array (assumed to be 400:50:700)
lambda_avg = [400, 450, 500, 550, 600, 650, 700];

% 2) LOAD LIGHT PENETRATION DATA FROM CSV
lp_data = readmatrix(parent_folder+'light_penetration_wavelength_depth_NOAA.csv');
wavelength_csv = lp_data(:,1);
depth_csv      = lp_data(:,2);
% (Optionally, smooth the depth data)
depth_csv_smoothed = depth_csv; % or use smoothdata(depth_csv, 'movmean', 15);

% Create an interpolation function over 400-700 nm.
lp_interp = @(lam) interp1(wavelength_csv, depth_csv_smoothed, lam, 'linear', 'extrap');

% 3) BUILD THE LIGHT PENETRATION UNDER-CURVE (COLORED)
lambda_bg = linspace(400,700,301);    % wavelengths from 400 to 700 nm.
depth_max = 225;                      % maximum depth in meters.
depth_bg  = linspace(0, depth_max, 201);  % depths from 0 to 225 m.

% Evaluate and clamp the penetration curve.
PenetrationCurve = lp_interp(lambda_bg);
PenetrationCurve = min(max(PenetrationCurve, 0), depth_max);

% Create a mask: for each (depth, wavelength) pair, mark if depth is shallower than the penetration depth.
[LambdaGrid, DepthGrid] = meshgrid(lambda_bg, depth_bg);
mask = DepthGrid <= repmat(PenetrationCurve, length(depth_bg), 1);

% Build a base-color array using the custom wavelength2rgb function.
baseColor = zeros(length(lambda_bg), 3);
for j = 1:length(lambda_bg)
    baseColor(j,:) = wavelength2rgb(lambda_bg(j));
end

% Build the colored under-curve image.
BG_image = ones(length(depth_bg), length(lambda_bg), 3);  % default white background.
for i = 1:length(depth_bg)
    for j = 1:length(lambda_bg)
        if mask(i,j)
            BG_image(i,j,:) = baseColor(j,:);
        end
    end
end

% Create an alpha mask for the under-curve (opacity 0.6 where mask is true).
alpha_data = double(mask) * 0.6;

% 4) CREATE BACKGROUND GRADIENT (WHITE-TO-BLACK)
nRows = 200; nCols = 300;
grad = repmat(linspace(1,0,nRows)', 1, nCols);  % vertical gradient: white at top, black at bottom.
gradRGB = repmat(grad, [1, 1, 3]);               % replicate for R, G, and B.
alpha_grad = repmat(linspace(0.8,0.5,nRows)', 1, nCols);  % alpha mask for gradient.

% 5) CREATE THE FIGURE WITH DUAL Y-AXES AND LAYERED BACKGROUNDS
figure(554); clf;
set(gcf, 'Color', 'w');

% --- BACKGROUND GRADIENT AXES ---
axBG = axes('Position', [0.13 0.11 0.775 0.815]);
hGrad = imagesc([400,700], [0,225], gradRGB);
set(axBG, 'YDir', 'normal');   % so depth increases upward.
axis(axBG, 'off');             % hide ticks on background.
set(hGrad, 'AlphaData', alpha_grad);  % apply alpha mask.
uistack(axBG, 'bottom');

% --- MAIN AXES ---
axMain = axes('Position', get(axBG, 'Position'));
yyaxis left
% Plot the colored under-curve for light penetration.
hImg = imagesc(lambda_bg, depth_bg, BG_image);
set(hImg, 'AlphaData', alpha_data);
set(gca, 'YDir', 'normal');
caxis([400 700]);
yticks([50 100 150 200]);
ylabel('Depth (m)', 'FontSize', 34, 'FontWeight', 'bold', 'Color', [0 0 0]);
ylim([0 depth_max]);
hold on;
% Overlay the smoothed penetration boundary curve.
plot(lambda_bg, PenetrationCurve, 'k-', 'LineWidth', 2);

yyaxis right
% Plot the average spectral discrimination (R) curve.
p_avg = plot(lambda_avg, R_wavelength, '-o', 'LineWidth', 3, 'MarkerSize', 9, ...
    'Color', [0.6 0 0], 'MarkerFaceColor', [0.6 0 0]);
ylabel('Spectral Discrimination Ratio (\eta)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.6 0 0]);
% Removed y-axis limit restriction to show full-scale data.
xlim([400 700]);
xlabel('Wavelength (nm)', 'FontSize', 34, 'FontWeight', 'bold');
grid on; grid minor;
title('Spectral Discrimination Ratio and Light Penetration in Seawater', ...
      'FontSize', 18, 'FontWeight', 'bold');

ax = gca;
ax.FontSize = 24;
ax.LineWidth = 1.2;
ax.YAxis(1).Color = [0 0 0];
ax.YAxis(2).Color = [0.6 0 0];
legend(p_avg, {'EVK4'}, 'Location', 'best', 'FontSize', 40);
hold off;

% LOCAL FUNCTION: Convert wavelength (nm) to RGB accurately
function rgb = wavelength2rgb(wavelength)
    % This function converts a wavelength (in nm) to an RGB triplet.
    gamma = 0.8;  % gamma correction factor.
    if wavelength >= 400 && wavelength < 440
        attenuation = 0.3 + 0.7*(wavelength - 400)/(440 - 400);
        R = ((-(wavelength - 440) / (440 - 400)) * attenuation) .^ gamma;
        G = 0.0;
        B = (1.0 * attenuation) .^ gamma;
    elseif wavelength >= 440 && wavelength < 490
        R = 0.0;
        G = ((wavelength - 440) / (490 - 440)) .^ gamma;
        B = 1.0;
    elseif wavelength >= 490 && wavelength < 510
        R = 0.0;
        G = 1.0;
        B = (-(wavelength - 510) / (510 - 490)) .^ gamma;
    elseif wavelength >= 510 && wavelength < 580
        R = ((wavelength - 510) / (580 - 510)) .^ gamma;
        G = 1.0;
        B = 0.0;
    elseif wavelength >= 580 && wavelength < 645
        R = 1.0;
        G = (-(wavelength - 645) / (645 - 580)) .^ gamma;
        B = 0.0;
    elseif wavelength >= 645 && wavelength <= 700
        attenuation = 0.3 + 0.7*(700 - wavelength)/(700 - 645);
        R = (1.0 * attenuation) .^ gamma;
        G = 0.0;
        B = 0.0;
    else
        R = 0; G = 0; B = 0;
    end
    rgb = [R, G, B];
end


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d, UnivariateSpline

output_filename = "spectral_resolution_simulation_frames"
data = loadmat('figures/rows_wavelengths_column_focal_length_intensitty_value_simulation_frames.mat')
wavelengths = [400, 450, 500, 550, 600, 650, 700, 750, 800]  # shape (n,)
focal_distances = np.linspace(1,10,200)  # shape (m,)
intensity = data['peak_all_group'].T  # shape (n, m)


# output_filename = "spectral_resolution_real_frames"
# data = loadmat('figures/rows_wavelengths_column_focal_length_intensitty_value_real_frames.mat')
# intensity = data['peak_all_group'] # shape (n, m)
# focal_distances = data['focal_all_group'].flatten()  # shape (m,)
# wavelengths = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]  # shape (n,)


# output_filename = "spectral_resolution_simulation_events"
# data = loadmat('figures/rows_wavelengths_column_focal_length_intensitty_value_simulation_events.mat')
# intensity = data['peak_all_group'].T # shape (n, m)
# focal_distances = data['focal_all_group'].flatten()  # shape (m,)
# wavelengths = [400, 450, 500, 550, 600, 650, 700, 750, 800]  # shape (n,)


# output_filename = "spectral_resolution_real_events"
# data = loadmat('figures/rows_wavelengths_column_focal_length_intensitty_value_real_events.mat')
# intensity = data['peak_all_group'].T # shape (n, m)
# focal_distances = data['focal_all_group'].flatten()  # shape (m,)
# wavelengths = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]  # shape (n,)


# Function to compute FWHM for a given intensity curve
def compute_fwhm(focal, inten):
    peak_idx = np.argmax(inten)
    peak_val = inten[peak_idx]
    half_max = peak_val / 2.0

    # Left side: search before the peak
    left_indices = np.where(inten[:peak_idx] <= half_max)[0]
    if left_indices.size > 0:
        left_index = left_indices[-1]
        # Linear interpolation between two points
        f_left = np.interp(half_max, [inten[left_index], inten[left_index+1]],
                           [focal[left_index], focal[left_index+1]])
    else:
        f_left = focal[0]

    # Right side: search after the peak
    right_indices = np.where(inten[peak_idx:] <= half_max)[0]
    if right_indices.size > 0:
        right_index = right_indices[0] + peak_idx
        f_right = np.interp(half_max, [inten[right_index-1], inten[right_index]],
                            [focal[right_index-1], focal[right_index]])
    else:
        f_right = focal[-1]

    fwhm = f_right - f_left
    return fwhm, focal[peak_idx]

# Compute FWHM and peak focal distance for each wavelength curve
fwhm_list = []
peak_focal_list = []
for i in range(len(wavelengths)):
    inten_curve = intensity[:,i]
    fwhm, peak_focal = compute_fwhm(focal_distances, inten_curve)
    fwhm_list.append(fwhm)
    peak_focal_list.append(peak_focal)

fwhm_array = np.array(fwhm_list)
peak_focal_array = np.array(peak_focal_list)

# window = 2  # number of points on each side of the current index (adjust as needed)
# n = len(peak_focal_array)
# dlam_df = np.empty(n)

# for i in range(n):
#     # Determine the indices for the local window
#     start = max(0, i - window)
#     end = min(n, i + window + 1)
#     # x: local focal distances, y: corresponding wavelengths
#     x_local = peak_focal_array[start:end]
#     y_local = np.array(wavelengths)[start:end]
#     if len(x_local) < 2:
#         dlam_df[i] = np.nan
#     else:
#         # Fit a line: y = m*x + b, where m is the slope.
#         p = np.polyfit(x_local, y_local, 1)
#         dlam_df[i] = p[0]

# # Now convert FWHM (in focal distance units) to spectral width Δλ (in nm)
# delta_lambda = dlam_df * fwhm_array

# # Compute spectral resolution R = λ/Δλ (using only the indices with valid derivative)
# valid = np.isfinite(delta_lambda) & (delta_lambda != 0)
# R = np.full_like(delta_lambda, np.nan)
# R[valid] = np.array(wavelengths)[valid] / delta_lambda[valid]


fwhm_array = np.array(fwhm_list)
peak_focal_array = np.array(peak_focal_list)

# Compute the derivative d(lambda)/df at the computed peak focal distances.
# Here we use np.gradient for a finite-difference derivative.
# Note: this assumes the f0's are ordered similarly to wavelengths.
dlam_df = np.gradient(wavelengths, peak_focal_array)

# Convert FWHM (in focal distance units) into spectral width Δλ (in nm)
delta_lambda = dlam_df * fwhm_array

# Compute the spectral resolution (resolving power): R = λ/Δλ (dimensionless)
R = wavelengths / delta_lambda


######################### Uncomment to save file as .mat #########################
# spectral_resolution = {"R": R, "wavelengths": wavelengths} 
# savemat(f'figures/{output_filename}.mat', spectral_resolution)

# Plot the spectral resolution vs. wavelength
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, R, marker='o', linestyle='-')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Resolution (λ/Δλ)')
plt.title('Spectral Discrimination Performance')
plt.grid(True)
plt.show()
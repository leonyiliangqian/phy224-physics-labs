import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Defining the sine model
def sine_model(t, a, b, c, phi):
    return a + b * t + c * np.sin(2 * np.pi * t - phi)


# Reading the data, not every single one is used
data = np.loadtxt("co2_mm_mlo.csv", comments='#', delimiter=',', skiprows=41)
year = data[:, 0]
month = data[:, 1]
decimal_date = data[:, 2]
average = data[:, 3]
deseasonalized = data[:, 4]
ndays = data[:, 5]
sdev = data[:, 6]
unc = data[:, 7]

# Collecting valid uncertainties, which are the ones that are greater than 0
valid_uncertainties = []
for num in unc:
    if num > 0:
        valid_uncertainties.append(num)

# Calculating the average uncertainty from the valid ones
average_uncertainty = sum(valid_uncertainties) / len(valid_uncertainties)

# Replacing invalid uncertainties, which are the ones that are smaller than
# or equal to 0, with the calculated average
unc_valid = []
for num in unc:
    if num <= 0:
        unc_valid.append(average_uncertainty)  # Replace invalid uncertainty
    else:
        unc_valid.append(num)  # Keep valid uncertainty


# Initial guesses for a, b, c, and phi based on the historical data
initial_guess = [300, 2, 10, 0]

# Fitting the model to the observations using valid uncertainties
popt, pcov = curve_fit(sine_model, decimal_date, average,
                       sigma=unc_valid, absolute_sigma=True, p0=initial_guess)

# Extracting fitted parameters and their uncertainties
a, b, c, phi = popt
param_uncertainties = np.sqrt(np.diag(pcov))

# Printing the fitted parameters with uncertainties
print(f"Fitted parameters:\n a = {a:.2f} ± {param_uncertainties[0]:.6}")
print(f" b = {b:.2f} ± {param_uncertainties[1]:.6f}")
print(f" c = {c:.2f} ± {param_uncertainties[2]:.6f}")
print(f" phi = {phi:.2f} ± {param_uncertainties[3]:.6f}")

# Modeling predictions using the fitted parameters
fitted_values = sine_model(decimal_date, a, b, c, phi)


# Plotting the original data and the fitted model
plt.figure()
plt.errorbar(decimal_date, average, yerr=unc_valid, fmt='-',
             label='Observed CO2 levels')
plt.plot(decimal_date, fitted_values, label='Sine Model Fit', color='orange')
plt.xlabel('Decimal Date')
plt.ylabel('CO2 levels (ppm)')
plt.title('CO2 Levels with Sine Model Fit')
plt.legend()
plt.show()

# Calculating residuals
residuals = average - fitted_values

# Plotting residuals
plt.figure()
plt.plot(decimal_date, residuals, 'x', label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Decimal Date')
plt.ylabel('Residuals (ppm)')
plt.title('Residuals of the Sine Model Fit')
plt.legend()
plt.show()


# Defining the improved model with quadratic growth,
# time-dependent amplitude, and multiple oscillations
def improved_model(t, a, b, d, c1_0, c1_1, phi1, c2_0, c2_1, phi2):
    # First sinusoidal component
    sin1 = (c1_0 + c1_1 * t) * np.sin(2 * np.pi * t - phi1)
    # Second sinusoidal component (higher frequency)
    sin2 = (c2_0 + c2_1 * t) * np.sin(4 * np.pi * t - phi2)
    growth = a + b * t + d * t**2
    return growth + sin1 + sin2


# New initial guesses for the parameters
initial_guess_new = [300, 2, 0.01, 10, 0.1, 0, 5, 0.05, 0]

# Fitting the improved model to the data
popt, pcov = curve_fit(improved_model, decimal_date, average,
                       sigma=unc_valid, absolute_sigma=True,
                       p0=initial_guess_new)

# Extracting fitted parameters
a, b, d, c1_0, c1_1, phi1, c2_0, c2_1, phi2 = popt

# Printing fitted parameters with uncertainties
param_uncertainties = np.sqrt(np.diag(pcov))
print(f"Fitted parameters:\n a = {a:.2f} ± {param_uncertainties[0]:.6f}")
print(f" b = {b:.2f} ± {param_uncertainties[1]:.6f}")
print(f" d = {d:.4f} ± {param_uncertainties[2]:.6f}")
print(f" c1_0 = {c1_0:.2f} ± {param_uncertainties[3]:.6f}")
print(f" c1_1 = {c1_1:.4f} ± {param_uncertainties[4]:.6f}")
print(f" phi1 = {phi1:.2f} ± {param_uncertainties[5]:.6f}")
print(f" c2_0 = {c2_0:.2f} ± {param_uncertainties[6]:.6f}")
print(f" c2_1 = {c2_1:.4f} ± {param_uncertainties[7]:.6f}")
print(f" phi2 = {phi2:.2f} ± {param_uncertainties[8]:.6f}")

# Modeling predictions using the fitted parameters
fitted_values = improved_model(decimal_date, a, b, d, c1_0,
                               c1_1, phi1, c2_0, c2_1, phi2)

# Plotting the original data and the fitted model
plt.errorbar(decimal_date, average, yerr=unc_valid, fmt='-',
             label='Observed CO2 levels')
plt.plot(decimal_date, fitted_values, label='Improved Model Fit',
         color='orange')
plt.xlabel('Decimal Date')
plt.ylabel('CO2 levels (ppm)')
plt.title('CO2 Levels with Improved Model Fit')
plt.legend()
plt.show()

# Calculating residuals
residuals = average - fitted_values

# Plotting residuals
plt.plot(decimal_date, residuals, 'x', label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Decimal Date')
plt.ylabel('Residuals (ppm)')
plt.title('Residuals of the Improved Model Fit')
plt.legend()
plt.show()



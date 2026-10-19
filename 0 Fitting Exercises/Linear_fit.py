# Setting up the program
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Defining the linear model function
def linear_model(x, a, b):
    """
    Linear model function: f(x) = ax + b, calculates the y-values based on
    the linear model.
    """
    return a * x + b


# Defining the chi squared function
def calculate_chi_squared(observed, expected, uncertainties):
    """
    Calculate the chi-squared statistic.
    """
    return np.sum(((observed - expected) / uncertainties) ** 2)


# Defining the reduced chi squared function
def calculate_reduced_chi_squared(chi_squared, num_data_points, num_parameters):
    """
    Calculate the reduced chi-squared statistic.
    """
    degrees_of_freedom = num_data_points - num_parameters
    return chi_squared / degrees_of_freedom


# Reading the data from the csv
data = np.loadtxt("co2_annmean_mlo.csv", comments='#', delimiter=',', skiprows=44)
year_num = data[:, 0]  # Extracting the year
emission_num = data[:, 1]  # Extracting CO2 measurements
uncertainty_num = data[:, 2]  # Extracting uncertainties

# Considering only the last 20 years of data
year_20 = year_num[-20:]
emission_20 = emission_num[-20:]
uncertainties_20 = uncertainty_num[-20:]

# Fit the model to the observations
popt, pcov = curve_fit(linear_model, year_20, emission_20,
                       sigma=uncertainties_20, absolute_sigma=True)


# Extracting the parameters from the curve fit model
a, b = popt
param_uncertainties = np.sqrt(np.diag(pcov))

# Model predictions using the best-fit parameters
fitted_emission = linear_model(year_20, a, b)

# Plot the dataset with error bars and the fitted model
plt.figure()
plt.errorbar(year_20, emission_20, yerr=uncertainties_20, fmt='o',
             label='Observed CO2 levels')
plt.plot(year_20, fitted_emission, label=f'Fitted model')
plt.xlabel('Year')
plt.ylabel('CO2 levels (ppm)')
plt.title('CO2 Levels and Linear Fit (Over the Last 20 Years)')
plt.legend()
plt.show()

# Characterize the fit
residuals = emission_20 - fitted_emission
chi_squared = calculate_chi_squared(emission_20, fitted_emission,
                                    uncertainties_20)
reduced_chi_squared = calculate_reduced_chi_squared(chi_squared,
                                                    len(emission_20), len(popt))
print(f"Reduced chi-squared: {reduced_chi_squared:.2f}")

# Plot the residuals
plt.figure()
plt.plot(year_20, residuals, 'o', label='Residuals')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Residuals (ppm)')
plt.title('Residuals of the Linear Fit')
plt.legend()
plt.show()

# Predict CO2 levels in 1960 and 2060
year_1960 = 1960
year_2060 = 2060
predicted_1960 = linear_model(year_1960, a, b)
predicted_2060 = linear_model(year_2060, a, b)
uncertainty_1960 = np.sqrt((param_uncertainties[0] * year_1960) ** 2 +
                           param_uncertainties[1] ** 2)
uncertainty_2060 = np.sqrt((param_uncertainties[0] * year_2060) ** 2 +
                           param_uncertainties[1] ** 2)


# Printing the results
print(f"Predicted CO2 level in 1960: {predicted_1960:.2f} ± {uncertainty_1960:.2f} ppm")
print(f"Predicted CO2 level in 2060: {predicted_2060:.2f} ± {uncertainty_2060:.2f} ppm")


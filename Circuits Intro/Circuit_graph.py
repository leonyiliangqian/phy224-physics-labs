# Setting up the program
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import ticker

# Data collected for the 100 Ohm resistor
input_voltage = np.array([0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000, 10.000]) #volts
voltage = np.array([0.0000, 0.9819, 1.9643, 2.9468, 3.929, 4.912, 5.893, 6.876, 7.859, 8.841, 9.824]) #volts
current = np.array([0.106, 9.800, 19.590, 29.408, 39.16, 49.01, 58.73, 68.58, 78.35, 88.12, 97.74]) #milliamp
voltage_lsds = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
current_lsds = np.array([0.001, 0.001, 0.001, 0.001, 0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01])


# Defining the linear model function
def linear_model(x, a, b):
    """
    Linear model function: f(x) = ax + b, calculates the y-values based on
    the linear model.
    """
    return a * x + b


def uncertainty_voltage(reading, least_sig):
    return reading * (0.05 / 100) + 5 * least_sig


def uncertainty_current(reading, least_sig):
    return reading * (0.2 / 100) + 5 * least_sig


# Defining the chi squared function
def calculate_chi_squared(observed, expected, uncertainties):
    """
    Calculate the chi-squared statistic.
    """
    return np.sum(((expected - observed) / uncertainties) ** 2)


# Defining the reduced chi squared function
def calculate_reduced_chi_squared(chi_squared, num_data_points, num_parameters):
    """
    Calculate the reduced chi-squared statistic.
    """
    degrees_of_freedom = num_data_points - num_parameters
    return chi_squared / degrees_of_freedom


current_uncertainty = uncertainty_current(current, current_lsds)
voltage_uncertainty = uncertainty_voltage(voltage, voltage_lsds)

p_opt, p_cov = curve_fit(linear_model, voltage, current, sigma=current_uncertainty, absolute_sigma=True)

a, b = p_opt
param_uncertainties = np.sqrt(np.diag(p_cov))
print(f"The value of a is {a:.2f} ± {param_uncertainties[0]:.2f}")
print(f"The value of b is {b:.2f} ± {param_uncertainties[1]:.2f}")
fitted_current = linear_model(voltage, a, b)

# Plotting the original data and the fitted model
fig, ax = plt.subplots()
ax.errorbar(voltage, current, yerr=current_uncertainty, fmt='o',
            label='Collected data')
ax.plot(voltage, fitted_current, label='Model Fit',
        color='orange')
ax.set_xlabel('Voltage')
ax.set_ylabel('Current')
ax.set_title('Current vs Voltage')
ax.grid()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.legend()
plt.show()


# Characterize the fit
residuals = current - fitted_current
chi_squared = calculate_chi_squared(current, fitted_current,
                                    current_uncertainty)
reduced_chi_squared = calculate_reduced_chi_squared(
    chi_squared, len(current), len(p_cov))
print(f"Chi_squared: {chi_squared:.2f}")
print(f"Reduced chi_squared: {reduced_chi_squared:.2f}")

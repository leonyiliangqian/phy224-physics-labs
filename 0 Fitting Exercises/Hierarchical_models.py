import numpy as np
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt


# Defining the quadratic growth model
def growth_model(t, a, b, d):
    return a + b * t + d * t**2


# Defining the periodic model
def periodic_model(t, c1, c2, phi):
    return c1 * np.sin(2 * np.pi * t - phi) + c2 * np.sin(4 * np.pi * t - phi)

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


# Fitting the growth model to the data
initial_growth_guess = [300, 2, 0.01]  # Initial guesses for growth
popt_growth, pcov_growth = curve_fit(growth_model, decimal_date, average, p0=initial_growth_guess)

# Extracting the fitted parameters for the growth model
a_growth, b_growth, d_growth = popt_growth

# Predicting growth component and calculate residuals (periodic component)
growth_fitted_values = growth_model(decimal_date, a_growth, b_growth, d_growth)
residuals_after_growth = average - growth_fitted_values

# Plotting growth model fit and residuals
plt.plot(decimal_date, average, '-', label='Observed CO2 levels')
plt.plot(decimal_date, growth_fitted_values, label='Growth Model Fit', color='orange')
plt.xlabel('Decimal Date')
plt.ylabel('CO2 levels (ppm)')
plt.title('Growth Model Fit')
plt.legend()
plt.show()

plt.plot(decimal_date, residuals_after_growth, 'x', label='Residuals (Periodic Component)')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Decimal Date')
plt.ylabel('CO2 Residuals (ppm)')
plt.title('Residuals after Subtracting Growth Model')
plt.legend()
plt.show()

# Fitting the periodic model to the residuals
initial_periodic_guess = [10, 5, 0]  # Initial guesses for periodic model
popt, pcov = curve_fit(periodic_model, decimal_date, residuals_after_growth, p0=initial_periodic_guess)

# Extracting the fitted parameters for the periodic model
c1_periodic, c2_periodic, phi_periodic = popt

# Predicting the periodic component
periodic_fitted_values = periodic_model(decimal_date, c1_periodic, c2_periodic, phi_periodic)

# Plotting the periodic model fit and the final residuals
plt.plot(decimal_date, residuals_after_growth, '-', label='Observed Periodic Component')
plt.plot(decimal_date, periodic_fitted_values, label='Periodic Model Fit', color='green')
plt.xlabel('Decimal Date')
plt.ylabel('CO2 Residuals (ppm)')
plt.title('Periodic Model Fit')
plt.legend()
plt.show()

# Final residuals after subtracting both growth and periodic components
final_residuals = residuals_after_growth - periodic_fitted_values

plt.plot(decimal_date, final_residuals, 'x', label='Final Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Decimal Date')
plt.ylabel('Residuals (ppm)')
plt.title('Final Residuals After Subtracting Growth and Periodic Models')
plt.legend()
plt.show()

# Combining growth and periodic components to predict the full model
full_model_fitted_values = growth_fitted_values + periodic_fitted_values

# Answering the 3 questions
# 1. Month that has the highest CO2 value
max_co2_index = np.argmax(full_model_fitted_values)
max_co2_value = full_model_fitted_values[max_co2_index]
max_co2_date = decimal_date[max_co2_index]
print(f"Month with the highest CO2 value: {max_co2_date:.2f} at {max_co2_value:.2f} ppm.")


# 2. CO2 level pass twice the pre-industrial level
def co2_double(t, a, b, d, c1, c2, phi):
    return growth_model(t, a, b, d) + periodic_model(t, c1, c2, phi) - 285 * 2


future_date = fsolve(co2_double, 2023, args=(a_growth, b_growth, d_growth, c1_periodic, c2_periodic, phi_periodic))[0]
print(f"CO2 level will pass 570 ppm around the year {future_date:.2f}.")

# 3. Future CO2 minimum exceeding the maximum of 2000
year_2000_mask = (decimal_date >= 2000) & (decimal_date < 2001)
co2_max_2000 = np.max(full_model_fitted_values[year_2000_mask])

def co2_min_exceeds_2000(t, a, b, d, c1, c2, phi, co2_max_2000):
    return growth_model(t, a, b, d) + periodic_model(t, c1, c2, phi) - co2_max_2000

future_min_exceeds_2000 = fsolve(co2_min_exceeds_2000, 2023, args=(a_growth, b_growth, d_growth, c1_periodic, c2_periodic, phi_periodic, co2_max_2000))[0]
print(f"The CO2 minimum will exceed the CO2 maximum from 2000 around the year {future_min_exceeds_2000:.2f}.")

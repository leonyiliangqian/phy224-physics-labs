import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Reading the data
data = np.genfromtxt("co2_temp_1000.csv", comments='#', delimiter=',', skip_footer=5)
year_num = data[:, 0]
temp_mean = data[:, 1]  # Extract the temperature mean
temp_std = data[:, 2]  # Extract the temperature standard deviation
co2_mean = data[:, 3]  # Extract the CO2 emission mean
co2_std = data[:, 4]  # Extract the CO2 emission stand deviation


# Defining the split year 1800 between the
# pre-industrial and fully industrial periods
split_year = 1800


# Splitting the data based on the defined periods
pre_industrial_years = year_num < split_year
industrial_years = year_num > split_year


# Pre-industrial period data
pre_industrial_year = year_num[pre_industrial_years]
pre_industrial_temp_mean = temp_mean[pre_industrial_years]
pre_industrial_temp_std = temp_std[pre_industrial_years]
pre_industrial_co2_mean = co2_mean[pre_industrial_years]
pre_industrial_co2_std = co2_std[pre_industrial_years]

# Industrial period data
industrial_year = year_num[industrial_years]
industrial_temp_mean = temp_mean[industrial_years]
industrial_temp_std = temp_std[industrial_years]
industrial_co2_mean = co2_mean[industrial_years]
industrial_co2_std = co2_std[industrial_years]

# Create subplots
plt.subplot(1, 2, 1)
plt.hist(pre_industrial_temp_mean, bins=20, density=True, histtype='bar', align='mid', color='green', label='Pre-industrial', alpha=0.75)
plt.hist(industrial_temp_mean, bins=20, density=True, histtype='bar', align='mid', color='orange', label='Industrial', alpha=0.75)
plt.xlabel('Surface Temperature Mean (°C)')
plt.ylabel('Density')
plt.title('Temperature Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(pre_industrial_co2_mean, bins=20, density=True, histtype='bar', align='mid', color='green', label='Pre-industrial', alpha=0.75)
plt.hist(industrial_co2_mean, bins=20, density=True, histtype='bar', align='mid', color='orange', label='Industrial', alpha=0.75)
plt.xlabel('CO2 Concentration (ppm)')
plt.ylabel('Density')
plt.title('CO2 Distribution')
plt.legend()

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)  # Increase the spacing between the two subplots

plt.show()

# Calculate means and standard deviations for both periods
pre_industrial_temp_mean_num = np.mean(pre_industrial_temp_mean)
pre_industrial_temp_std_num = np.std(pre_industrial_temp_mean)
industrial_temp_mean_num = np.mean(industrial_temp_mean)
industrial_temp_std_num = np.std(industrial_temp_mean)

pre_industrial_co2_mean_num = np.mean(pre_industrial_co2_mean)
pre_industrial_co2_std_num = np.std(pre_industrial_co2_mean)
industrial_co2_mean_num = np.mean(industrial_co2_mean)
industrial_co2_std_num = np.std(industrial_co2_mean)

print("Pre-Industrial Period:")
print(f"Temperature Mean: {pre_industrial_temp_mean_num:.2f} °C, Standard Deviation: {pre_industrial_temp_std_num:.2f} °C")
print(f"CO2 Mean: {pre_industrial_co2_mean_num:.2f} ppm, Standard Deviation: {pre_industrial_co2_std_num:.2f} ppm\n")

print("Industrial Period:")
print(f"Temperature Mean: {industrial_temp_mean_num:.2f} °C, Standard Deviation: {industrial_temp_std_num:.2f} °C")
print(f"CO2 Mean: {industrial_co2_mean_num:.2f} ppm, Standard Deviation: {industrial_co2_std_num:.2f} ppm\n")


def log_model(C, a, b):
    return a * np.log(C) + b



# Filter out rows with NaN or inf values before fitting the model
valid_indices = ~np.isnan(co2_mean) & ~np.isnan(temp_mean) & np.isfinite(co2_mean) & np.isfinite(temp_mean)

# Only use valid data for the fit
C = co2_mean[valid_indices]
T = temp_mean[valid_indices]

# Fit the model
popt, pcov = curve_fit(log_model, C, T)
a, b = popt
a_unc, b_unc= np.sqrt(np.diag(pcov))

# Print the values of a and b
print(f"Fitted value of a: {a:.2f} ± {a_unc:.2f}")
print(f"Fitted value of b: {b:.2f} ± {b_unc:.2f}")


# Step 4: Prediction for doubling of CO2 from pre-industrial levels
doubling_co2 = 2 * pre_industrial_co2_mean
predicted_temp_change = log_model(doubling_co2, a, b)


# Plot the fit
plt.subplot(2, 1, 1)
plt.plot(C, T, 'x', label='Data', color='blue')
plt.plot(C, log_model(C, a, b), color='red', label=f'Fitted model')
plt.xlabel('CO2 Concentration (ppm)')
plt.ylabel('Surface Temperature Mean (°C)')
plt.title('CO2 vs Temperature Regression')
plt.legend()

# Calculate residuals
residuals = T - log_model(C, a, b)

# Plot residuals
plt.subplot(2, 1, 2)
plt.plot(C, residuals, 'x', label='Residuals', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('CO2 Concentration (ppm)')
plt.ylabel('Residuals (°C)')
plt.title('Residuals of the Fit')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the prediction
plt.figure()
plt.plot(year_num[valid_indices], log_model(C, a, b), 'x', label='Predicted Temperature')
plt.plot(year_num[valid_indices], T, 'x', label='Observed Temperature', color='orange')
plt.xlabel('Year')
plt.ylabel('Surface Temperature Mean (°C)')
plt.title('Predicted vs Observed Temperature')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend()
plt.show()

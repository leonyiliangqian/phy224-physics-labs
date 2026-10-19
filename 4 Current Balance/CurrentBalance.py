# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

# %%
# Constants
M_0 = 4 * np.pi * 10**-7
l = .285 # m
h0 = .108 # m
h0 = .250 - h0 # m
L = 1.826 # m
L_B = .215 # m

# Read dataset
data = pd.read_csv('./Current Balance Data.csv')

# Extracting the columns
mass = data['Mass (mg)'].to_numpy()
current = data['Current (AAC)'].to_numpy()
reading = data['Reading on the ruler (mm)'].to_numpy()

# Uncertainties
mass = unp.uarray(mass, 2/100 * mass) / 1e6 # ±2% uncertainty & convert to kg
reading = h0 - unp.uarray(reading, 1) / 1000 # ±1 mm & convert to meters
current_range = 20
current = unp.uarray(current, .5/100 * current + .3/100 * current_range) # ±(0.5% reading + 0.3% of range)

# %%
# Calculate the distance between the wires
# Take the reading from the max weight and 0 current as the rest position
# and calculate the distance between the wires relative to that

r0 = h0 - 161 / 1000 # m
theta0 = 1/2 * np.arctan(r0 / L) # rad

# Calculate the separation for each reading
thetas = 1/2 * unp.arctan(reading / L) - theta0
d = 2 * L_B * unp.sin(thetas / 2)

data = pd.DataFrame({'Mass': unp.std_devs(mass), 'Current': current, 'Distance': d, 'Angles': thetas})

# Group by mass
grouped = data.groupby('Mass')

# Fit the data
def fit(I, a, b, c):
    return a * I**2 + b * I + c

fit_params = []

# Plot the fit
fig, ax = plt.subplots()
for name, group in grouped:
    I = group['Current']
    d = group['Distance']
    popt, pcov = curve_fit(fit, unp.nominal_values(I), unp.nominal_values(d), sigma=unp.std_devs(d), absolute_sigma=True)
    a, b, c = unc.correlated_values(popt, pcov)
    x = np.linspace(0, 10, 100)
    ax.plot(x, fit(x, *popt), label='Fit')
    print(f'{name} kg:')
    print(f'a = {a:.2EP}')
    print(f'b = {b:.2EP}')
    print(f'c = {c:.2EP}')
    print(f'χ^2 = {np.sum((fit(unp.nominal_values(I), *popt) - unp.nominal_values(d))**2 / unp.std_devs(d)**2)}')
    print(f'χ^2/ν = {np.sum((fit(unp.nominal_values(I), *popt) - unp.nominal_values(d))**2 / unp.std_devs(d)**2) / (len(I) - 3)}\n')
    fit_params.append((a, b, c))
    ax.errorbar(unp.nominal_values(I), unp.nominal_values(d), xerr=unp.std_devs(I), yerr=unp.std_devs(d), fmt='o', label=f'{name} kg')
ax.set_title('Distance vs Current')
ax.set_xlabel('I (A)')
ax.set_ylabel('d (m)')
ax.legend()
plt.show()

# Plot the residuals
fig, ax = plt.subplots()
for name, group in grouped:
    I = group['Current']
    d = group['Distance']
    a, b, c = fit_params[int(name * 1000)]
    a = unc.nominal_value(a)
    b = unc.nominal_value(b)
    c = unc.nominal_value(c)
    ax.errorbar(unp.nominal_values(I), fit(unp.nominal_values(I), a, b, c) - unp.nominal_values(d), yerr=unp.std_devs(d), fmt='o', label=f'{name} kg')
ax.set_title('Residuals for distance vs current')
ax.set_xlabel('I (A)')
ax.set_ylabel('Residual (m)')
ax.legend()
plt.show()

# %%
I_const_deflection = []

# Solve for the currents when the distance is 0.005m
print('Currents when the distance is 0.005m')
masses = np.array([name for name, group in grouped])
for i, (a, b, c) in enumerate(fit_params):
    a = unc.nominal_value(a)
    b = unc.nominal_value(b)
    c = unc.nominal_value(c)
    I = max(np.roots([a, b, c - 0.005]))
    I_const_deflection.append(I)
    print(f'Current required for deflection of 0.005m for {masses[i]} kg: {I:.2f} A')
I_const_deflection = np.array(I_const_deflection)
print()

# Fit the data
def fit2(m, a, b):
    return a * m + b

popt, pcov = curve_fit(fit2, masses, I_const_deflection**2, sigma=1 / 100 * I_const_deflection**2, absolute_sigma=True) # ±1% uncertainty
a, b = unc.correlated_values(popt, pcov)
x = np.linspace(0, 2e-6, 100)
plt.plot(x, fit2(x, *popt), label='Fit')
plt.errorbar(masses, I_const_deflection**2, yerr=1 / 100 * I_const_deflection**2, fmt='o')
plt.title('Current Squared vs Mass for 0.005 m deflection')
plt.xlabel('m (kg)')
plt.ylabel('$I^2$ $(A^2)$')
plt.legend()
plt.show()

print('Fit for I^2 = a * m + b:')
print(f'a = {a:.2EP}')
print(f'b = {b:.2EP}')
print(f'χ^2 = {np.sum((fit2(masses, *popt) - I_const_deflection**2)**2 / (1 / 100 * I_const_deflection**2)**2)}')
print(f'χ^2/ν = {np.sum((fit2(masses, *popt) - I_const_deflection**2)**2 / (1 / 100 * I_const_deflection**2)**2) / (len(masses) - 2)}')

# Calculate Mu_0
mu_0 = 1/a
print(f'Mu_0 = {mu_0:.2E} T m/A')



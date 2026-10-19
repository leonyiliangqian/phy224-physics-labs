import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import ticker


# Defining the expected model function for fitting
def expected_model(v_0, t, r, c):
    return v_0 * np.exp(-t / (r * c))


# Reading the data from the csv exported by the cute ass oscilloscope
data = np.loadtxt("scope_3.csv", dtype=np.float64, comments='#', delimiter=',')
time = data[:,0]
response_voltage = data[:,1]
input_voltage = data[:,2]


# Plotting the dataset
plt.figure()
plt.plot(time, response_voltage, label=f'Response Voltage')
plt.plot(time, input_voltage, label=f'Input Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Graph of Voltage versus Time')
plt.legend()
plt.show()



print(time[-1]-time[0])

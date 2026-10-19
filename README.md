# PHY224 Practical Physics Labs (University of Toronto)

Data analysis for the PHY224 experimental physics labs, written in Python (NumPy, SciPy, Matplotlib). Each lab collects real measurements, fits a physical model with `scipy.optimize.curve_fit`, propagates uncertainties, and checks the fit with residual plots and reduced chi-squared.

Lab partner: Terry Chan.

| Lab | What it measures | Methods |
|---|---|---|
| [Fitting Exercises](0%20Fitting%20Exercises) | Mauna Loa CO2 trend and forecast | Linear, quadratic, periodic and hierarchical models; prediction to 2060 |
| [Circuits Intro](Circuits%20Intro) | RC circuit response from oscilloscope data | Exponential fits |
| [Light Bulb](Lightbulb) | Tungsten filament I-V curve | Power-law fit on linear and log axes |
| [Capacitor](Capacitor) | RC charging and discharging | Manual vs oscilloscope measurement, exponential decay fit |
| [Diode](Diode) | Forward and reverse diode I-V curve | Exponential I-V fit, with and without series resistor |
| [Electron Charge to Mass](Electron%20Charge%20to%20Mass) | e/m of the electron in Helmholtz coils | Fixed-current and fixed-voltage fits, Earth field correction |
| [Pulses in Cables](3%20Pulses%20in%20Cables%204) | Pulse speed and reflections in a coaxial line | LC transmission-line model, delay fits |
| [Current Balance](4%20Current%20Balance) | Force between current-carrying wires | Linear fit to recover mu_0 |

Lab write-ups are in [Lab Reports](Lab%20Reports).

## Run

```bash
pip install numpy scipy matplotlib pandas jupyter
jupyter notebook
```

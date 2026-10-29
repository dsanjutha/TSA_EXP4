# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# Date: 29.10.26



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Upload dataset
uploaded = files.upload()

# Step 2: Load dataset
file_name = next(iter(uploaded))
data = pd.read_csv(BytesIO(uploaded[file_name]))

print("Dataset loaded successfully!")
print(data.head())

# Step 3: Prepare the car price data
data.columns = data.columns.str.strip()
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data = data.dropna(subset=['year', 'price'])

# Select a sample for faster model fitting
data = data.sample(
    n=min(5000, len(data)),
    random_state=42
)

# Arrange data by year
data = data.sort_values('year')
X = data['price'].reset_index(drop=True)

# Set figure size
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

# Step 4: Visualize original data
plt.plot(X.to_numpy())
plt.title('Original Car Price Data')
plt.xlabel('Observation')
plt.ylabel('Price')
plt.show()

# Step 5: ACF and PACF of original data
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(X, lags=35, ax=axes[0])
axes[0].set_title('Original Data ACF')

plot_pacf(X, lags=35, ax=axes[1])
axes[1].set_title('Original Data PACF')

plt.tight_layout()
plt.show()

# Step 6: Fit ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

print("\nARMA(1,1) Model")
print(arma11_model.summary())

# Step 7: Simulate ARMA(1,1) process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.xlim([0, 500])
plt.show()

# Step 8: ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1, lags=35)
plt.title('ACF of ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1, lags=35)
plt.title('PACF of ARMA(1,1)')
plt.show()

# Step 9: Fit ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.arparams[0]
phi2_arma22 = arma22_model.arparams[1]
theta1_arma22 = arma22_model.maparams[0]
theta2_arma22 = arma22_model.maparams[1]

print("\nARMA(2,2) Model")
print(arma22_model.summary())

# Step 10: Simulate ARMA(2,2) process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.xlim([0, 500])
plt.show()

# Step 11: ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2, lags=35)
plt.title('ACF of ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2, lags=35)
plt.title('PACF of ARMA(2,2)')
plt.show()

print("\nARMA model fitting completed successfully!")
```

OUTPUT:

Original data:



<img width="862" height="618" alt="image" src="https://github.com/user-attachments/assets/45038a7a-5700-4cbe-ba33-2d8b067056bd" />




Partial Autocorrelation:



<img width="824" height="286" alt="image" src="https://github.com/user-attachments/assets/45e85ad2-7c12-4ad6-a0e7-dd45bd6a80f7" />



Autocorrelation:



<img width="820" height="280" alt="image" src="https://github.com/user-attachments/assets/15d2bbc6-4181-41a9-9977-f884cdb6a31f" />


SIMULATED ARMA(1,1) PROCESS:



<img width="700" height="361" alt="image" src="https://github.com/user-attachments/assets/15c374b4-38da-429b-bf59-129fee2335de" />






<img width="758" height="410" alt="image" src="https://github.com/user-attachments/assets/7764bb20-06e1-440a-80c4-c6b3a0b729ba" />






Partial Autocorrelation:




<img width="762" height="398" alt="image" src="https://github.com/user-attachments/assets/7e5e1409-a14b-4a16-96cf-8d8854ea33f4" />




Autocorrelation:



<img width="753" height="402" alt="image" src="https://github.com/user-attachments/assets/8e11d31c-19b8-4f38-bac9-c4efd62c2868" />



SIMULATED ARMA(2,2) PROCESS:



<img width="836" height="442" alt="image" src="https://github.com/user-attachments/assets/3d4466c4-7951-423a-b397-0a4fea916b8a" />





<img width="769" height="414" alt="image" src="https://github.com/user-attachments/assets/7cc413a0-cb79-4297-8ccd-cd501bf12f18" />






Partial Autocorrelation:





<img width="768" height="394" alt="image" src="https://github.com/user-attachments/assets/6bc4c5aa-0f94-4173-a2d7-0e52b58e0276" />





Autocorrelation:




<img width="758" height="402" alt="image" src="https://github.com/user-attachments/assets/4ec04555-f70e-4ac2-9633-f796a4b16f7f" />




RESULT:



Thus, a python program is created to fir ARMA Model successfully.

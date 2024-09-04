import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Load data while skipping the first row
truth_data = pd.read_csv('0.5_truth_flattened.csv', header=None, skiprows=1).values.flatten()
trkn_data = pd.read_csv('0.5_trkn_flattened.csv', header=None, skiprows=1).values
trkp_data = pd.read_csv('0.5_trkp_flattened.csv', header=None, skiprows=1).values
emcal_data = pd.read_csv('0.5_emcal_flattened.csv', header=None, skiprows=1).values
hcal_data = pd.read_csv('0.5_hcal_flattened.csv', header=None, skiprows=1).values

# Prepare input and output matrices
XX = np.hstack([trkn_data, trkp_data, emcal_data, hcal_data])
Y = truth_data

# Normalize data
scaler = StandardScaler()
XX_normalized = scaler.fit_transform(XX)

# Define neural network model with adjusted hyperparameters
model = MLPRegressor(hidden_layer_sizes=(64, 32, 16),  # Reduced hidden layers
                     activation='relu',
                     solver='adam',
                     alpha=0.001,
                     batch_size=1024,  # Smaller batch size
                     learning_rate_init=0.001,
                     max_iter=100,
                     shuffle=True,
                     verbose=True,
                     validation_fraction=0.1,
                     early_stopping=True,
                     n_iter_no_change=10)

# Train the network on the entire dataset
model.fit(XX_normalized, Y)

# Predict with the neural network using the entire dataset
Y_pred = model.predict(XX_normalized)

# Save Y_pred variable as a CSV file
pd.DataFrame(Y_pred).to_csv('Y_pred_full.csv', index=False, header=False)

# Display success message
print('Y_pred has been successfully saved as a CSV file.')

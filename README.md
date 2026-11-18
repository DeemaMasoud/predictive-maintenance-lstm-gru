# Predictive Maintenance Using LSTM and GRU

A deep learning application for predicting the Remaining Useful Life (RUL) of turbofan engines.

## Dataset
NASA C-MAPSS FD001 dataset.

## Models
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## Features
- Upload CSV, TXT, or Excel files
- Select individual engine IDs
- Predict RUL using LSTM and GRU
- Explain GRU predictions using SHAP
- Visualize temporal attributions using Integrated Gradients

## Model Performance

| Metric | LSTM | GRU |
|---|---|---|
| MAE | 16.357 | 16.306 |
| RMSE | 25.221 | 23.911 |
| R² | 0.632 | 0.669 |

## Technologies
Python, TensorFlow, Keras, Flask, Scikit-learn, SHAP, HTML, and CSS.

## Run Locally
1. Install the packages in requirements.txt.
2. Run `python app.py`.
3. Open http://127.0.0.1:5000.

## Input Data
The application requires at least 30 operating cycles and 17 selected input features for each prediction.

The models were trained on FD001. Predictions on other C-MAPSS subsets or real industrial equipment have not been validated.
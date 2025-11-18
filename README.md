# Microsoft Stock Price Prediction (RNN)

## Project Overview

This project uses deep learning to predict the closing price of Microsoft (MSFT) stock. It implements and compares two types of Recurrent Neural Networks (RNNs) — **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** — to forecast the next day's closing price based on a sequence of the previous 20 days' closing prices.

The task is a "many-to-one" time-series forecasting problem.

## Dataset

The project uses the `Microsoft_Stock.csv` dataset, which contains daily stock data (Date, Open, High, Low, Close, Volume). For this prediction task, only the `Close` price is used as the feature.

## Methodology

1.  **Data Preprocessing:**
    * The `Close` price data is extracted from the dataset.
    * The data is normalized using `MinMaxScaler` to a range of `(-1, 1)`. This is a crucial step for the stability and performance of RNNs.

2.  **Sequence Creation:**
    * A sliding window approach is used to transform the time-series data into supervised learning samples.
    * A sequence of **20** consecutive closing prices is used as the input (X) to predict the closing price of the following day (y).

3.  **Train-Test Split:**
    * The sequenced data is split into a training set (80%) and a testing set (20%).
    * To maintain the chronological order of the time-series, the split is done by index, not randomly.

4.  **Data Loading:**
    * PyTorch `TensorDataset` and `DataLoader` are used to create batches for efficient training and evaluation.

## Models

Two RNN architectures are defined, trained, and compared:

1.  **LSTM (`net1`):** A 2-layer LSTM network with a hidden size of 32, followed by a fully-connected layer.
2.  **GRU (`net2`):** A 2-layer GRU network with a hidden size of 32, followed by a fully-connected layer.

Both models are trained for 50 epochs using the **Mean Squared Error (MSE)** loss function and the **Adam** optimizer.

## Results

The GRU model demonstrated significantly better performance on the test set compared to the LSTM model.

* **LSTM Test MSE:** `0.0183`
* **GRU Test MSE:** `0.0031`

The final plots visualize the actual prices against the predicted prices from both models on the training and test sets.

### Test Set Predictions

### Train Set Predictions

## Requirements

This project requires the following Python libraries:

* `pandas`
* `numpy`
* `torch` (PyTorch)
* `torchmetrics`
* `scikit-learn` (for MinMaxScaler)
* `matplotlib`

## How to Use

1.  Ensure you have the required libraries installed.
2.  Make sure the `Microsoft_Stock.csv` file is available (you may need to update the file path in the notebook).
3.  Run the Jupyter Notebook (`Microsoft stock (RNN).ipynb`) cells sequentially.
4.  The models will be trained, evaluated, and the prediction plots will be saved to the directory as `.png` files.

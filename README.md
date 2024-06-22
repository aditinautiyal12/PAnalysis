# Crypto Price Prediction Project

## Vision

The goal of this project is to develop a system that leverages cryptocurrency price data to predict future price trends using machine learning techniques. By utilizing historical data retrieved daily from a public API, the system will train a machine learning model to forecast price movements for the next two days. The predictions will then be compared against actual prices to visualize trends and evaluate the model's accuracy.

## Objectives

1. **Data Collection:**

   - Retrieve daily cryptocurrency price data from a reliable API (e.g., WazirX API).
   - Store historical price data for analysis and training.

2. **Machine Learning Model Development:**

   - Use the last 5 days of historical price data to train a machine learning model.
   - Implement regression or time series forecasting techniques suitable for predicting cryptocurrency prices.
   - Evaluate and fine-tune the model to optimize prediction accuracy.

3. **Prediction and Visualization:**
   - Predict cryptocurrency prices for the next 2 days based on the trained model.
   - Visualize the predicted prices alongside actual prices in a trend graph.
   - Generate insights into the model's performance and the accuracy of its predictions.

## Project Components

- **Data Retrieval:** Python script to fetch daily cryptocurrency price data from the WazirX API.
- **Data Processing and Model Training:** Jupyter notebook or Python script to preprocess the data, train the machine learning model, and evaluate its performance.
- **Prediction and Visualization:** Python script or web application to generate predictions, create trend graphs, and display comparative analysis of predicted vs. actual prices.

## Usage

1. **Setup:**

   - Clone the repository.
   - Install necessary Python libraries (`pandas`, `requests`, `scikit-learn`, etc.) using `pip`.

2. **Data Collection:**

   - Run the data retrieval script (`fetch_data.py`) to fetch and store daily cryptocurrency price data.

3. **Model Training and Prediction:**

   - Use the Jupyter notebook (`crypto_price_prediction.ipynb`) to preprocess data, train the machine learning model, and make predictions.

4. **Visualization:**
   - Implement the visualization script (`visualize.py`) to create trend graphs showing actual vs. predicted cryptocurrency prices.

## Dependencies

- Python 3.x
- Pandas
- Requests
- Scikit-learn (for machine learning models)
- Matplotlib or Plotly (for visualization)

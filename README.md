# WISDM Activity Recognition - Machine Learning Pipeline

This project implements a complete machine learning pipeline for the WISDM (Wireless Sensor Data Mining) Activity Recognition dataset. The goal is to predict human activities (Jogging, Walking, Standing, Sitting, Upstairs, Downstairs) using smartphone accelerometer data.

## Project Structure
- `WISDM_ar_v1.1_raw.txt`: Raw sensor data.
- `pipeline.py`: Main script for cleaning, preprocessing, EDA, and modeling.
- `plots/`: Directory containing EDA and evaluation visualizations.
- `pipeline_output.log`: Detailed logs of the execution.

## Steps Involved

### 1. Data Cleaning
- **Parsing**: Handled the raw text format, including semi-colons and potential malformed lines.
- **Handling Duplicates**: 66,587 duplicate rows were identified and removed.
- **Missing Values**: Rows with incomplete sensor readings or labels were discarded during parsing.

### 2. Preprocessing
- **Categorical Encoding**: The target variable `activity` was encoded into numerical labels using `LabelEncoder`.
- **Standardization**: Feature scaling was applied to the accelerometer axes (X, Y, Z) using `StandardScaler` to ensure zero mean and unit variance, which helps models like Logistic Regression converge faster.
- **Data Splitting**: The dataset was split into 80% training and 20% testing sets, with stratification to maintain class balance.

### 3. Exploratory Data Analysis (EDA)
The following visualizations were generated in the `plots/` directory:
- `activity_distribution.png`: Shows the frequency of each activity.
- `accel_distributions.png`: Histograms of X, Y, and Z accelerometer values.
- `accel_by_activity.png`: Boxplots showing the variation of acceleration for different activities.
- `correlation_matrix.png`: Heatmap showing correlations between sensor axes.

### 4. Model Selection and Training
Two models were compared:
- **Logistic Regression**: Used as a baseline linear classifier.
- **Random Forest Classifier**: A robust non-linear ensemble model.

**Random Forest** performed significantly better due to its ability to capture complex patterns in the sensor data.

### 5. Best Fitting Model Results
- **Chosen Model**: Random Forest Classifier
- **Accuracy**: 60.62% (on raw per-record classification)
- **Evaluation Metrics**:
    - High precision/recall for "Sitting" and "Standing".
    - Moderate performance for "Jogging" and "Walking".
    - Lower performance for "Upstairs" and "Downstairs" due to their similar accelerometer profiles when looking at individual records.

*Note: For higher accuracy (90%+), time-series windowing and feature engineering (FFT, mean, variance over windows) are typically required.*

## How to Run
1. Ensure the required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the pipeline:
   ```bash
   python pipeline.py
   ```
3. Check the `plots/` folder for visualizations and `pipeline_output.log` for the detailed report.

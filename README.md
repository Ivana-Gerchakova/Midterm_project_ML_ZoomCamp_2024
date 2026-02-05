# Midterm_project_ML_ZoomCamp_2024


<img src="images/zoomcamp.jpg" />

## Project Overview: 
This project analyzes the New York City Airbnb dataset and builds a machine-learning model to predict rental prices. Completed as part of the ML Zoomcamp course from DataTalksClub. 
The project includes detailed steps in data preparation, exploratory data analysis (EDA), model training (including neural networks), model tuning, deployment, and reproducibility, aligned with project requirements.

<img src="images/dataset-cover.jpg" />

## Problem Description: 
Airbnb prices can vary significantly based on various factors like location, room type, and availability. This project aims to predict the rental price based on these factors, assisting hosts in optimizing pricing strategies and helping users find cost-effective options.

This model can be used to:
- Help new Airbnb hosts set competitive prices based on historical trends.
- Provide insights into key factors affecting pricing, enabling better business decisions.
- Assist customers in identifying cost-effective options.

## Dataset: 

- **Source**: [Kaggle - New York City Airbnb dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

- **Description**: 
   This dataset includes detailed information about Airbnb listings in New York City, featuring attributes such as:
   - Listing details: Price, minimum nights, availability.
   - Location: Longitude, latitude, and neighborhood.
   - Host information: Number of reviews, reviews per month.
   
- **Instructions**: Download the dataset directly from Kaggle using the provided link or follow the instructions within this repository for loading the data.

## Technologies Used:
- **Programming Languages**: Python
- **Libraries**: Pandas, Scikit-learn, XGBoost, Flask, Docker

  ## Exploratory Data Analysis (EDA)

The exploratory data analysis (EDA) phase provided critical insights into the structure, distribution, and relationships within the dataset. 
Here’s a detailed breakdown of the steps performed:

### 1. **Target Variable Analysis (Price)**
- The target variable (`price`) exhibits a highly skewed distribution, with most listings priced under $200.
- A boxplot analysis highlighted several outliers (listings priced over $1,000), which were either excluded or capped for better model performance.
- Logarithmic transformation of `price` was considered to normalize its distribution but was excluded to maintain interpretability.

**Visualization:**
- A histogram showcasing the distribution of `price` was plotted, emphasizing the skewness.
- Boxplots were used to identify and handle outliers effectively.

### 2. **Feature Analysis**
Key features were analyzed to understand their impact on price:

#### **Room Type**
- Listings categorized as "Entire home/apt" dominate the higher price range.
- "Shared room" and "Private room" listings are generally clustered in the lower price range.

**Visualization:**
- A bar plot displaying the average price by room type demonstrated a clear difference in pricing trends.

#### **Location**
- `longitude` and `latitude` values revealed clustering of listings by boroughs (e.g., Manhattan, Brooklyn).
- Higher prices were concentrated in Manhattan and parts of Brooklyn.

**Visualization:**
- A scatter plot of `longitude` vs. `latitude`, color-coded by price, visually confirmed the influence of location.

#### **Minimum Nights**
- Listings with higher `minimum_nights` values showed a slight reduction in average price, suggesting they cater to long-term stays.
- Outliers in `minimum_nights` (e.g., 365 days) were handled appropriately.

### 3. **Correlation Analysis**
- A correlation heatmap revealed significant relationships between features:
  - `availability_365` and `number_of_reviews` showed moderate correlation.
  - `room_type` and `neighborhood_group` were identified as strong predictors for price based on group statistics.

**Visualization:**
- A heatmap visualized the pairwise correlations among features, providing clarity on feature selection.


### 4. **Missing Data**
- Missing values were identified in the `reviews_per_month` column (~20%).
- Imputation was performed using the median for this column, ensuring no data loss.
- No significant missing data was found in other critical features.

### 5. **Additional Insights**
- Listings with higher review counts and positive reviews tend to have more competitive pricing.
- Seasonal availability (based on `availability_365`) highlighted patterns for long-term rental properties.


### Visuals and Findings:
- All visualizations and findings are available in the accompanying notebook, showcasing:
  - Histograms of numeric variables.
  - Scatter plots for feature relationships.
  - Correlation heatmaps for feature selection.

### Conclusion:
The EDA phase helped identify key patterns and relationships, confirming the importance of features like `room_type`, `location`, and `availability_365` in determining rental prices. This analysis laid the foundation for selecting and tuning predictive models.

## Model Training

In this project, multiple machine-learning models were trained and evaluated to predict Airbnb rental prices. Below is an overview of the models used and the insights gained from their performance:

### 1. **Models Trained**
The following models were trained to understand their ability to predict rental prices:
- **Linear Regression:** A simple, interpretable model used as a baseline for comparison.
- **Decision Tree Regressor:** Captures non-linear relationships but is prone to overfitting.
- **Random Forest Regressor:** An ensemble method that improves prediction accuracy by averaging multiple decision trees.
- **Gradient Boosting:** A boosting technique that combines weak learners to enhance predictive performance.
- **XGBoost:** A high-performance implementation of Gradient Boosting, optimized for speed and scalability.

### 2. **Insights from Model Performance**
- **Best Performing Models:**
  - **XGBoost** and **Gradient Boosting** were the top-performing models, achieving the lowest RMSE and highest R² scores.
  - These models effectively captured the variance in the data while minimizing prediction errors.

- **Ensemble Models:**
  - Ensemble methods like **Random Forest**, **XGBoost**, and **Gradient Boosting** significantly outperformed simpler models. This highlights the power of ensemble techniques in improving accuracy and reducing overfitting.

- **Linear Regression:**
  - Linear Regression struggled to capture complex relationships in the data, resulting in the highest RMSE and lowest R² scores.

- **Decision Tree:**
  - The Decision Tree model performed better than Linear Regression but lacked the generalization power of ensemble models.

### 3. **Model Selection**
After comparing the models based on metrics like RMSE and R², **XGBoost** was selected as the final model for deployment due to:
- Its ability to balance accuracy and generalization.
- Its robust error minimization capabilities.

**Gradient Boosting** is a close alternative, demonstrating comparable performance, and may be considered for future deployment scenarios.

### 4. **Conclusion**
The training process demonstrated the importance of testing multiple models and leveraging ensemble methods for complex datasets. The XGBoost model's superior performance makes it the recommended choice for deployment, with Gradient Boosting as a viable alternative.

### 5. **Neural Networks**
- A simple feedforward neural network was trained using TensorFlow/Keras.
- The network architecture included multiple Dense layers with activation functions and dropout regularization to prevent overfitting.
- Despite testing various configurations, the neural network did not outperform ensemble methods.

### 6. **Model Comparison**
- All models were evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
- The comparison allowed the selection of the best-performing model for deployment.

### 7. **Model Selection**
- Based on the evaluation metrics, an ensemble-based model (e.g., XGBoost) was chosen as the final model due to its superior performance.
- The selected model was saved for deployment using the `pickle` library:
  ```python
  import pickle
  with open("final_model.pkl", "wb") as f:
      pickle.dump(best_model, f)

### Exporting Notebook to Script

The logic from the Jupyter Notebook was exported into separate Python scripts to improve modularity and reproducibility. 
Below is the structure of the exported scripts:

1. **`data_preparation.py`:** Handles data loading and cleaning processes.
2. **`eda.py`:** Performs exploratory data analysis (EDA), including visualizations and feature importance analysis.
3. **`model_training.py`:** Trains multiple machine learning models and saves the best-performing model.
4. **`model_evaluation.py`:** Evaluates the models and provides insights into their performance.

To execute these scripts:
1. Prepare the data using `data_preparation.py`:
   ```bash
   python data_preparation.py

2. Perform EDA using  `eda.py`:
   ```bash
   python eda.py

3. Train the final model using `model_training.py`:
   ```bash
   python model_training.py

4. Evaluate the model using `model_evaluation.py`:
   ```bash
   python model_evaluation.py

## Reproducibility

The project has been designed to ensure full reproducibility. All requirements have been met, including:

- **Data Accessibility:** Clear instructions are provided for downloading the dataset from Kaggle, ensuring all users can access the data.
- **Independent Scripts:** Each workflow stage (data preparation, EDA, model training, evaluation) is modularized into separate Python scripts that can be executed independently.
- **Reproducible Workflow:** Detailed instructions are provided to run all scripts step-by-step, allowing anyone to replicate the results without issues.

This ensures the project is fully reproducible and adheres to the required standards.

## Model Deployment

The project includes the deployment of the trained model as a REST API using the **Flask** framework. This deployment allows users to send input data via a POST request and receive predictions in real time.

### Deployment Details
- **Framework**: Flask
- **Environment**: Developed in **Google Colab**, but the setup can be replicated locally or in a cloud environment.
- **Endpoint**: `/predict`
- **Input Format**: JSON
- **Output Format**: JSON containing the model's predictions.

### Google Colab Deployment

To use Google Colab, follow these steps:

1. **Install the flask-ngrok library**:
   ```bash
   pip install flask-ngrok
   pip install flask

## Dependency and Environment Management

To set up the required dependencies for this project, follow these steps:

`!pip freeze > requirements.txt` 

`pandas==2.2.2`

`numpy==1.26.4`

`scikit-learn==1.5.2`

`matplotlib==3.8.0`

`seaborn==0.13.2`

### Containerization

This project includes containerization using Docker to ensure a consistent and portable environment for running the application. 

#### 1. **Dockerfile**
A `Dockerfile` is included in the project, which specifies the dependencies and environment setup. The `requirements-Docker.txt` file contains the following dependencies:

`pandas==1.3.3`

`numpy==1.21.2`

`scikit-learn==0.24.2`

`xgboost==1.4.2`

`prettytable==2.4.0`

`matplotlib==3.4.3`

`seaborn==0.11.2`

`tensorflow==2.6.0`

`xgboost==2.1.2`

### Final Conclusion

This project successfully demonstrates the application of machine learning techniques to predict Airbnb rental prices in New York City. By leveraging a detailed exploratory data analysis (EDA) process and testing multiple models, the following milestones were achieved:

- **Data Preparation and Analysis:** The dataset was cleaned, missing values were imputed, and important features were analyzed. Key insights, such as the influence of location and room type on pricing, were identified.
- **Model Training and Evaluation:** Multiple models, including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and a neural network, were trained and evaluated. After extensive testing, **XGBoost** was selected as the final model due to its superior performance metrics.
- **Reproducibility and Deployment:** The project adheres to best practices for reproducibility by providing modularized scripts and detailed instructions. The final model was deployed as a REST API using Flask, allowing for real-time predictions.
- **Containerization:** A Dockerfile was created to ensure a portable and consistent environment for running the application.

#### Use Cases
This model can be used by:
1. Airbnb hosts to set competitive prices based on historical data and trends.
2. Customers to identify cost-effective accommodation options.
3. Researchers or businesses to gain insights into factors influencing Airbnb pricing.

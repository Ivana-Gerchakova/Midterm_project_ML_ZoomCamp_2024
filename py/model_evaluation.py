# -*- coding: utf-8 -*-
"""Project 25.11.24.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ncPLUke6sM0GPQs_DRLpUcsbXdosQZMr

### MIDTERM PROJECT

### LIBRARIES :
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

"""### READ THE DATASET :"""

df = pd.read_csv('New York City Airbnb.csv')
df.head()

df.tail()

"""### EDA"""

df.info()

"""### MISSING VALUES"""

df.isnull().sum()

"""### HANDLING COLUMNS WITH MISSING VALUES"""

unique_reviews_df = pd.DataFrame(df['reviews_per_month'].unique(), columns=['unique_reviews_per_month'])

print(unique_reviews_df)

median_value = df['reviews_per_month'].median()

df['reviews_per_month'] = df['reviews_per_month'].fillna(median_value)

"""### DROPPING UNNECESSARY COLUMNS"""

df.drop(columns=['name','host_id', 'host_name', 'last_review'], inplace=True)

df.duplicated().sum()

df.shape

df.describe().T

df.columns

df.nunique()

cat_cols=df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("----------------------------------")
print("Numerical Variables:")
print(num_cols)

df.dtypes

"""### COLUMNS WITH A STANDARD DEVIATION"""

std_values = df[num_cols].std()
std_values

"""### CHECK"""

df.info()

"""### MIN-MAX VALUES"""

for col in df.select_dtypes(include=np.number).columns:
    print(f'{col}: Min = {df[col].min()}, Max = {df[col].max()}')

"""### Outliers were detected and removed based on the interquartile range (IQR) method to ensure robust model performance."""

# Detecting outliers in 'price' column
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Removing outliers
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

"""### ANALYSIS OF TARGET VARIABLES"""

print(df['price'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x=df['price'], color='yellow')
plt.title("Price Distribution Boxplot")
plt.show()

df.hist(bins=30, figsize=(15, 10))
plt.show()

plt.figure(figsize=(15, 6))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.xticks(rotation=45)
plt.show()

col_with_no_std = df.select_dtypes(include=np.number).columns.tolist()

num_cols = len(col_with_no_std)

fig, axes = plt.subplots(nrows=(num_cols // 2) + (num_cols % 2), ncols=2, figsize=(8, (num_cols // 2 + 1) * 3))

axes = axes.flatten()

for i, col in enumerate(col_with_no_std):
    sns.histplot(df[col], kde=True, ax=axes[i], color='yellow', bins=30, shrink=0.9)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

correlation_matrix = df[col_with_no_std].corr().round(2)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, cmap="BrBG", annot=True)
plt.show()

data = df['neighbourhood_group'].value_counts()
explode = (0.1,) * len(data)

plt.figure(figsize=(10, 10))

data.plot(kind='pie', autopct="%0.1f%%", explode=explode)

plt.title('Distribution of neighbourhood_group')
plt.ylabel('')
plt.show()

plt.figure(figsize=(10,10))
y= sns.barplot(data=df, x = 'neighbourhood_group', y='price',color = 'orange')
y.bar_label(y.containers[0])
plt.title('Neighbourhood group by price')
plt.show()

sns.countplot(x='room_type', data=df, palette='pastel', order=df['room_type'].value_counts().index, hue='room_type')

plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    data=df,
    x='longitude',
    y='latitude',
    hue='price',
    palette='coolwarm',
    size='price',
    sizes=(10, 200),
    alpha=0.6
)

plt.title('Airbnb Locations in NYC by Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


plt.legend(title='Price', loc='upper right', bbox_to_anchor=(1.15, 1))

plt.show()

"""### FEATURE IMPORTANCE"""

numerical_features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365']

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix[['price']].sort_values(by='price', ascending=False), annot=True, cmap="coolwarm")
plt.title("Correlation of Numerical Features with Price")
plt.show()

neighbourhood_price_range = df.groupby('neighbourhood_group')['price'].agg(['min', 'max', 'mean', 'median']).sort_values(by='mean', ascending=False)
print("Price Ranges by Neighbourhood Group:")
print(neighbourhood_price_range)

room_type_price_range = df.groupby('room_type')['price'].agg(['min', 'max', 'mean', 'median']).sort_values(by='mean', ascending=False)
print("\nPrice Ranges by Room Type:")
print(room_type_price_range)

"""Numerical Features: Some numerical features show a strong correlation with price, making important for prediction.

Categorical Features:(neighbourhood_group) and (room_type) are significant factors impacting price, providing valuable insights into the pricing structure across various segments.
These insights help inform the selection of key features for building models to predict Airbnb prices.

### TRAIN/VALIDATION/TEST SPLIT
"""

df_temp, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_temp, test_size=0.25, random_state=1)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

target_train = df_train['price'].values
target_validation = df_val['price'].values
target_test = df_test['price'].values

df_train.drop(columns=['price'], inplace=True)
df_val.drop(columns=['price'], inplace=True)
df_test.drop(columns=['price'], inplace=True)

"""### DICTVECTORIZER"""

train_records = df_train.to_dict(orient='records')
vectorizer = DictVectorizer(sparse=True)
X_train_vectorized = vectorizer.fit_transform(train_records)

validation_records = df_val.to_dict(orient='records')
X_validation_vectorized = vectorizer.transform(validation_records)

test_records = df_test.to_dict(orient='records')
X_test_vectorized = vectorizer.transform(test_records)

print(f'Shape of vectorized training data: {X_train_vectorized.shape}')
print(f'Shape of vectorized validation data: {X_validation_vectorized.shape}')
print(f'Shape of vectorized test data: {X_test_vectorized.shape}')

"""### FEATURE IMPORTANCE"""

model = DecisionTreeRegressor(max_depth=1, random_state=1)
model.fit(X_train_vectorized, target_train)

feature_importance = model.feature_importances_
important_feature_index = feature_importance.argmax()
important_feature = vectorizer.feature_names_[important_feature_index]

print(f'The feature used for splitting the data is: {important_feature}\n')

"""### FUNCTION FOR EVALUATION METRICS"""

def evaluate_model(predictions, target):
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE: {mae:.3f}')
    print(f'R²: {r2:.3f}\n')

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_vectorized)
X_validation_scaled = scaler.transform(X_validation_vectorized)

"""### MODEL TRAINING WITH ONE MODEL, NO PARAMETER TUNING."""

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, target_train)

linear_val_predictions = linear_model.predict(X_validation_scaled)
print('Linear Regression Model with Scaled Data:')
evaluate_model(linear_val_predictions, target_validation)

"""Linear Regression Model meaning it explains only 16% of the variance in the target variable. Its show struggles with variance, likely due to the model's simplicity in capturing complex relationships.

### TRAINED MULTIPLE MODELS (Desision tree, RandomForestRegressor, XGBRegressor) WITH PARAMETER TUNING
"""

dt_params = {'max_depth': [1, 5, 10, None], 'min_samples_split': [2, 10, 20]}
dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=1), dt_params, cv=3)
dt_grid.fit(X_train_vectorized, target_train)
best_dt_model = dt_grid.best_estimator_

dt_val_predictions = best_dt_model.predict(X_validation_vectorized)
print('Decision Tree Model (Best Parameters):')
evaluate_model(dt_val_predictions, target_validation)

"""Decision Tree Model meaning it explains very little of the variance in the data. This model is less accurate, likely due to overfitting on certain patterns and lacking generalization."""

rf_params = {'n_estimators': [10, 20, 50], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=1, n_jobs=-1), rf_params, cv=2)
rf_grid.fit(X_train_vectorized, target_train)
best_rf_model = rf_grid.best_estimator_

rf_val_predictions = best_rf_model.predict(X_validation_vectorized)
print('Random Forest Model (Best Parameters):')
evaluate_model(rf_val_predictions, target_validation)

"""Random Forest Model performs slightly better with an R², indicating moderate explanatory power and variance handling. It show improved accuracy due to ensemble learning, which reduces overfitting."""

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=1)
xgb_model.fit(X_train_vectorized, target_train)

xgb_val_predictions = xgb_model.predict(X_validation_vectorized)
print('XGBoost Model Evaluation:')
evaluate_model(xgb_val_predictions, target_validation)

"""XGBoost Model has the best R² score, RMSE, and a low MAE, among the regression models, indicating it captures patterns slightly better. This is due to its boosted ensemble technique, which corrects errors iteratively and enhances predictive performance.

#### NEW FEATURES AND MODELS
"""

df['latitude_longitude'] = df['latitude'] * df['longitude']
df['reviews_per_month_ratio'] = df['number_of_reviews'] / (df['minimum_nights'] + 1)

"""### GRADIENT BOOSTING MODEL WITH TUNING"""

gb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=1), gb_params, cv=3, scoring='r2')
gb_grid.fit(X_train_vectorized, target_train)
best_gb_model = gb_grid.best_estimator_

gb_val_predictions = best_gb_model.predict(X_validation_vectorized)
print('Gradient Boosting Model (Best Parameters):')
evaluate_model(gb_val_predictions, target_validation)

"""Gradient Boosting Model shows performance similar to the Decision Tree, with an R², suggesting limited accuracy. This model show that it also struggled to generalize well."""

def evaluate_model(predictions, target):
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

evaluation_results = {
    "Linear Regression": evaluate_model(linear_val_predictions, target_validation),
    "Decision Tree": evaluate_model(dt_val_predictions, target_validation),
    "Random Forest": evaluate_model(rf_val_predictions, target_validation),
    "XGBoost": evaluate_model(xgb_val_predictions, target_validation),
    "Gradient Boosting": evaluate_model(gb_val_predictions, target_validation)
}

model_metrics = {
    "Model": list(evaluation_results.keys()),
    "RMSE": [result['RMSE'] for result in evaluation_results.values()],
    "R2 Score": [result['R2'] for result in evaluation_results.values()],
    "MAE": [result['MAE'] for result in evaluation_results.values()]
}

metrics_df = pd.DataFrame(model_metrics)

print(metrics_df)

"""### Interpretation:

1. **Best Performing Models**: XGBoost and Gradient Boosting are the top-performing models, achieving the lowest RMSE and highest R² scores. Both models capture the data's variance effectively while maintaining lower prediction errors.

2. **Ensemble Models**: Random Forest, XGBoost, and Gradient Boosting outperform simpler models (Linear Regression and Decision Tree), which demonstrates the power of ensemble techniques in improving accuracy and reducing overfitting.

3. **Linear Regression**: This model has the highest RMSE and lowest R² score, indicating it struggles to capture complex relationships in the data.

4. **Decision Tree**: Though it performs better than Linear Regression, it is prone to overfitting and lacks the generalization power of ensemble models.

### Conclusion:

The XGBoost model is the recommended choice for deployment due to its balance of accuracy, error minimization, and generalization capabilities. Gradient Boosting is a close alternative, demonstrating comparable performance.

### TRAINING MULTIPLE VARIATIONS OF NEURAL NETWORKS WITH TUNED PARAMETERS
"""

from keras.callbacks import EarlyStopping

def create_nn_model(dropout_rate=0.2, learning_rate=0.001, layer_size=64, additional_layers=False):
    model = Sequential()
    model.add(Dense(layer_size, activation='relu', input_shape=(X_train_vectorized.shape[1],)))
    model.add(Dropout(dropout_rate))

    if additional_layers:
        model.add(Dense(layer_size // 2, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

variations = [
    {'dropout_rate': 0.2, 'learning_rate': 0.001, 'layer_size': 64, 'additional_layers': False},
    {'dropout_rate': 0.3, 'learning_rate': 0.001, 'layer_size': 64, 'additional_layers': True},
    {'dropout_rate': 0.2, 'learning_rate': 0.0005, 'layer_size': 128, 'additional_layers': True},
    {'dropout_rate': 0.4, 'learning_rate': 0.0001, 'layer_size': 32, 'additional_layers': False},
]

trained_models = []
nn_predictions = []

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for i, params in enumerate(variations):
    print(f'\nTraining Neural Network Variation {i+1} with params: {params}')
    model_nn = create_nn_model(**params)

    model_nn.fit(
        X_train_vectorized,
        target_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_validation_vectorized, target_validation),
        callbacks=[early_stopping],
        verbose=1
    )

    trained_models.append(model_nn)
    nn_val_predictions = model_nn.predict(X_validation_vectorized).flatten()
    nn_predictions.append(nn_val_predictions)
    print(f'Neural Network Model Variation {i+1} completed.')

"""### EVALUATING OF NEURAL NETWORKS"""

def evaluate_nn_model(predictions, target):
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE: {mae:.3f}')
    print(f'R²: {r2:.3f}\n')

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

nn_results = []
for i, predictions in enumerate(nn_predictions):
    print(f'\nEvaluating Neural Network Variation {i+1}')
    metrics = evaluate_nn_model(predictions, target_validation)
    nn_results.append({
        "Variation": i + 1,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"]
    })

nn_results_df = pd.DataFrame(nn_results)
print(nn_results_df)

plt.figure(figsize=(10, 6))
plt.bar(nn_results_df['Variation'], nn_results_df['RMSE'], color='skyblue')
plt.xlabel('Variation')
plt.ylabel('RMSE')
plt.title('Neural Network Performance Comparison')
plt.show()

"""Neural networks showed negative R² values across all variations, indicating that they performed worse than a baseline model. This suggests that the networks struggled with high variance and failed to effectively capture meaningful relationships in the data.

The RMSE values were generally higher for neural networks compared to traditional regression models, further emphasizing their lower accuracy in this context.

Key Insight: These neural network models may require further hyperparameter tuning, feature engineering, or alternative architectures to match or surpass the performance of simpler regression models. This highlights that simpler models can sometimes outperform more complex ones, especially when working with structured or limited datasets.

### SAVING FILE IN PICKLE - TRAINING MODELS
"""

import pickle

models = {"Linear Regression": linear_model,
          "Decision Tree": best_dt_model,
          "Random Forest": best_rf_model,
          "XGBoost": xgb_model,
          "Gradient Boosting": best_gb_model}

for model_name, model in models.items():
    with open(f"{model_name.replace(' ', '_').lower()}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"{model_name} trained and saved as '{model_name.replace(' ', '_').lower()}_model.pkl'")

for i, model_nn in enumerate(trained_models):
    model_path = f"neural_network_model_variation_{i+1}.h5"
    model_nn.save(model_path)
    print(f"Neural Network Variation {i+1} saved as '{model_path}'")

"""### SAVING FILE IN PICKLE - EVALUATIONS MODELS  """

model_files = ["linear_regression_model.pkl",
               "decision_tree_model.pkl",
               "random_forest_model.pkl",
               "xgboost_model.pkl",
               "gradient_boosting_model.pkl"]

evaluation_results = {}

def evaluate_model(predictions, target):
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}

for model_file in model_files:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_validation_vectorized)
    model_name = model_file.replace("_model.pkl", "").title()
    evaluation_results[model_name] = evaluate_model(predictions, target_validation)

nn_evaluation_results = {}

def evaluate_nn_model(predictions, target):
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}

for i, predictions in enumerate(nn_predictions):
    model_name = f"Neural Network Variation {i+1}"
    nn_evaluation_results[model_name] = evaluate_nn_model(predictions, target_validation)

evaluation_results.update(nn_evaluation_results)

with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(evaluation_results, f)

!!pip install Flask

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    input_data = data['input']

    prediction = model.predict([input_data])

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

!pip freeze > requirements.txt

dockerfile_content = """
# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Run the model service (Flask or FastAPI as an example)
CMD ["python", "predict.py"]
"""

with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

from google.colab import files
files.download('Dockerfile')
files.download('requirements.txt')












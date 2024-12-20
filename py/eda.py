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

"""
Numerical Features: Some numerical features show a strong correlation with price, making important for prediction.
"""


### Categorical Features:(neighbourhood_group) and (room_type) are significant factors impacting price, providing valuable insights into the pricing structure across various segments.
### These insights help inform the selection of key features for building models to predict Airbnb prices.

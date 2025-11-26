import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv('insurance.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'insurance.csv' not found. Please ensure the file is in the same directory.")
        return

    print("\n--- Data Preparation ---")
    print(f"Original shape: {df.shape}")
    
    missing_values = df.isnull().sum().sum()
    print(f"Missing values found: {missing_values}")
    if missing_values > 0:
        df = df.dropna()
        print("Dropped rows with missing values.")

    le = LabelEncoder()
    df_encoded = df.copy()
    
    df_encoded['smoker_code'] = df_encoded['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    
    df_encoded['sex_code'] = le.fit_transform(df_encoded['sex'])
    df_encoded['region_code'] = le.fit_transform(df_encoded['region'])

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], bins=30, kde=True)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    print("Plot 1 (Histogram) generated.")

    print("\n--- Unsupervised Learning: K-Means Clustering ---")

    X_cluster = df[['bmi', 'charges']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.tight_layout()
    plt.show()
    print("Plot 2 (Elbow Plot) generated.")

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"Clustering complete with k={optimal_k}.")

    plt.figure(figsize=(10, 8))
    
    corr_cols = ['age', 'bmi', 'children', 'charges', 'smoker_code']
    corr_matrix = df_encoded[corr_cols].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    print("Plot 3 (Correlation Heatmap) generated.")

    print("\n--- Supervised Learning: Linear Regression (Fitting) ---")

    X_reg = df[['age']]
    y_reg = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_reg)

    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    r2 = r2_score(y_reg, y_pred)
    
    print(f"Linear Model Equation: Charges = {slope:.2f} * Age + {intercept:.2f}")
    print(f"R-squared Score: {r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], df['charges'], alpha=0.5, label='Actual Data')
    plt.plot(df['age'], y_pred, linewidth=2, label=f'Fit: y={slope:.1f}x + {intercept:.0f}')
    
    plt.title('Linear Fitting: Age vs Insurance Charges')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Plot 4 (Linear Fitting) generated.")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bmi', y='charges', hue='Cluster', palette='viridis', style='smoker')
    plt.title('Clusters: BMI vs Charges')
    plt.tight_layout()
    plt.show()
    print("Bonus Plot (Cluster Visualization) generated.")

if __name__ == "__main__":
    main()

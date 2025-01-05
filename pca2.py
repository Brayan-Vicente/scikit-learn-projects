import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def plot_variance_ratio(pca, title="Explained Variance Ratio"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 
             'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load and examine the data
    dt_heart = pd.read_csv('./data/heart.csv')
    print("Dataset Preview:")
    print(dt_heart.head())
    print("\nDataset Shape:", dt_heart.shape)

    # Prepare features and target
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # Scale the features
    dt_features_scaled = StandardScaler().fit_transform(dt_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        dt_features_scaled, 
        dt_target, 
        test_size=0.3, 
        random_state=42
    )

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Configure and fit PCA
    n_components = 3
    pca = PCA(n_components=n_components)
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)

    # Fit and transform data using both PCA methods
    pca_train = pca.fit_transform(X_train)
    pca_test = pca.transform(X_test)

    ipca_train = ipca.fit_transform(X_train)
    ipca_test = ipca.transform(X_test)

    # Plot explained variance ratio
    plot_variance_ratio(pca, "PCA Explained Variance Ratio")

    # Print explained variance ratio for each component
    print("\nExplained variance ratio by component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {ratio:.3f}")

    # Train and evaluate models
    logistic = LogisticRegression(solver='lbfgs', max_iter=1000)

    # PCA model
    logistic.fit(pca_train, y_train)
    pca_score = logistic.score(pca_test, y_test)
    print(f"\nPCA Score: {pca_score:.3f}")

    # IPCA model
    logistic_ipca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_ipca.fit(ipca_train, y_train)
    ipca_score = logistic_ipca.score(ipca_test, y_test)
    print(f"IPCA Score: {ipca_score:.3f}")
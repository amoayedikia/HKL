"""
Basic usage example for HKL (Harmony search Kullback-Leibler) algorithm
"""

from hkl import HKL
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    print("HKL Feature Selection - Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Generate an imbalanced dataset
    print("\n1. Generating imbalanced dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=200,
        n_informative=30,
        n_redundant=20,
        n_repeated=10,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],  # 90% majority, 10% minority
        flip_y=0.02,
        random_state=42
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Class distribution: {np.bincount(y)} (Ratio: {np.max(np.bincount(y))/np.min(np.bincount(y)):.1f}:1)")
    
    # Step 2: Apply HKL feature selection
    print("\n2. Applying HKL feature selection...")
    hkl = HKL(
        HMS=30,      # Harmony Memory Size
        HMCR=0.9,    # Harmony Memory Consideration Rate
        PAR=0.3,     # Pitch Adjustment Rate
        beta=0.7,    # Weight for fitness function
        NI=100       # Number of iterations (reduced for example)
    )
    
    # Fit HKL and transform the data
    X_selected = hkl.fit_transform(X, y)
    selected_features = hkl.get_selected_features()
    
    print(f"   Original features: {X.shape[1]}")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Feature reduction: {(1 - len(selected_features)/X.shape[1])*100:.1f}%")
    
    # Step 3: Compare classification performance
    print("\n3. Comparing classification performance...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_sel, X_test_sel, _, _ = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train on original features
    print("\n   a) Using all features:")
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train)
    X_test_scaled = scaler1.transform(X_test)
    
    svm1 = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm1.fit(X_train_scaled, y_train)
    y_pred1 = svm1.predict(X_test_scaled)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred1).ravel()
    sensitivity1 = tp / (tp + fn)
    specificity1 = tn / (tn + fp)
    gmean1 = np.sqrt(sensitivity1 * specificity1)
    
    print(f"      Sensitivity: {sensitivity1:.3f}")
    print(f"      Specificity: {specificity1:.3f}")
    print(f"      G-mean: {gmean1:.3f}")
    
    # Train on selected features
    print("\n   b) Using HKL-selected features:")
    scaler2 = StandardScaler()
    X_train_sel_scaled = scaler2.fit_transform(X_train_sel)
    X_test_sel_scaled = scaler2.transform(X_test_sel)
    
    svm2 = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm2.fit(X_train_sel_scaled, y_train)
    y_pred2 = svm2.predict(X_test_sel_scaled)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred2).ravel()
    sensitivity2 = tp / (tp + fn)
    specificity2 = tn / (tn + fp)
    gmean2 = np.sqrt(sensitivity2 * specificity2)
    
    print(f"      Sensitivity: {sensitivity2:.3f}")
    print(f"      Specificity: {specificity2:.3f}")
    print(f"      G-mean: {gmean2:.3f}")
    
    # Step 4: Show top features by KL divergence
    print("\n4. Top 10 features by KL divergence:")
    feature_importances = hkl.get_feature_importances()
    top_indices = np.argsort(feature_importances)[::-1][:10]
    
    for i, idx in enumerate(top_indices):
        print(f"   Feature {idx}: KL divergence = {feature_importances[idx]:.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
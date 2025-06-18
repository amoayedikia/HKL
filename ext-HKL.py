import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import time


class HKLExtended:
    """
    Extended Harmony search Kullback-Leibler (HKL) algorithm with additional utilities
    for practical applications and analysis.
    """
    
    def __init__(self, HMS=30, HMCR=0.9, PAR=0.3, beta=0.7, alpha=1.0, NI=150, 
                 n_bins=50, random_state=None, verbose=True):
        self.HMS = HMS
        self.HMCR = HMCR
        self.PAR = PAR
        self.beta = beta
        self.alpha = alpha
        self.NI = NI
        self.n_bins = n_bins
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Algorithm state
        self.HM = None
        self.fitness_values = None
        self.feature_kl_divergence = None
        self.best_harmony = None
        self.best_fitness = -np.inf
        self.D_max_KL = 0
        
        # Tracking convergence
        self.fitness_history = []
        self.kl_history = []
        self.feature_stability = []
        self.selected_features_history = []
        
    def sigmoid(self, x):
        """Sigmoid activation function with numerical stability"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    def calculate_kl_divergence_continuous(self, feature_data, y):
        """Calculate KL divergence for continuous features"""
        # Get class indices
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("HKL is designed for binary classification")
        
        # Determine minority and majority classes
        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        minority_values = feature_data[minority_idx]
        majority_values = feature_data[majority_idx]
        
        # Adaptive discretization
        n_bins = min(self.n_bins, int(np.sqrt(len(feature_data))))
        
        # Use quantiles for better bin boundaries
        all_values = np.concatenate([minority_values, majority_values])
        bins = np.quantile(all_values, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicate bins
        
        if len(bins) < 3:  # Handle constant features
            return 0.0
        
        bins[0] -= 1e-10
        bins[-1] += 1e-10
        
        # Compute histograms
        minority_hist, _ = np.histogram(minority_values, bins=bins)
        majority_hist, _ = np.histogram(majority_values, bins=bins)
        
        # Apply Laplace smoothing
        alpha_smooth = 0.1
        minority_hist = (minority_hist + alpha_smooth) / (len(minority_values) + alpha_smooth * len(bins) - 1)
        majority_hist = (majority_hist + alpha_smooth) / (len(majority_values) + alpha_smooth * len(bins) - 1)
        
        # Calculate KL divergence
        kl_div = 0
        for i in range(len(minority_hist)):
            if minority_hist[i] > 0 and majority_hist[i] > 0:
                kl_div += minority_hist[i] * np.log(minority_hist[i] / majority_hist[i])
                
        return kl_div
    
    def calculate_feature_kl_divergence(self, X, y):
        """Calculate KL divergence for all features with progress tracking"""
        n_features = X.shape[1]
        kl_divergences = np.zeros(n_features)
        
        if self.verbose:
            print("Calculating KL divergence for features...")
        
        for i in range(n_features):
            if self.verbose and i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{n_features} features")
            
            feature_data = X[:, i]
            
            # Check if feature has variance
            if np.var(feature_data) < 1e-10:
                kl_divergences[i] = 0
                continue
            
            # Check if feature is categorical
            unique_ratio = len(np.unique(feature_data)) / len(feature_data)
            
            if unique_ratio < 0.05:  # Categorical
                kl_divergences[i] = self.calculate_kl_divergence_categorical(feature_data, y)
            else:  # Continuous
                kl_divergences[i] = self.calculate_kl_divergence_continuous(feature_data, y)
        
        if self.verbose:
            print(f"  Completed KL divergence calculation for {n_features} features")
            print(f"  Mean KL divergence: {np.mean(kl_divergences):.4f}")
            print(f"  Max KL divergence: {np.max(kl_divergences):.4f}")
        
        return kl_divergences
    
    def calculate_kl_divergence_categorical(self, feature_data, y):
        """Calculate KL divergence for categorical features"""
        # Get class information
        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        unique_values = np.unique(feature_data)
        
        # Calculate probability mass functions
        kl_div = 0
        alpha_smooth = 0.1
        
        for val in unique_values:
            minority_count = np.sum(feature_data[minority_idx] == val)
            majority_count = np.sum(feature_data[majority_idx] == val)
            
            # Apply Laplace smoothing
            p_min = (minority_count + alpha_smooth) / (len(minority_idx) + alpha_smooth * len(unique_values))
            p_maj = (majority_count + alpha_smooth) / (len(majority_idx) + alpha_smooth * len(unique_values))
            
            if p_min > 0:
                kl_div += p_min * np.log(p_min / p_maj)
                
        return kl_div
    
    def calculate_gmean(self, y_true, y_pred):
        """Calculate G-mean with handling for edge cases"""
        if len(np.unique(y_true)) != 2:
            raise ValueError("G-mean requires binary classification")
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return np.sqrt(sensitivity * specificity)
    
    def evaluate_harmony(self, harmony, X, y):
        """Evaluate harmony with cross-validation for stability"""
        selected_features = np.where(harmony == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0
        
        # Use stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X[:, selected_features], y, test_size=0.3, 
            random_state=self.random_state, stratify=y
        )
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVM with balanced class weights
            svm = SVC(kernel='rbf', random_state=self.random_state, 
                     class_weight='balanced', probability=True)
            svm.fit(X_train_scaled, y_train)
            
            # Predict and calculate G-mean
            y_pred = svm.predict(X_test_scaled)
            gmean = self.calculate_gmean(y_test, y_pred)
            
            return gmean
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Evaluation failed with error: {e}")
            return 0.0
    
    def calculate_fitness(self, harmony, X, y):
        """Calculate fitness with tracking"""
        gmean = self.evaluate_harmony(harmony, X, y)
        
        selected_features = np.where(harmony == 1)[0]
        if len(selected_features) > 0:
            total_kl = np.sum(self.feature_kl_divergence[selected_features])
        else:
            total_kl = 0
        
        if total_kl > self.D_max_KL:
            self.D_max_KL = total_kl
        
        if self.D_max_KL > 0:
            fitness = self.beta * gmean + (1 - self.beta) * (total_kl / self.D_max_KL)
        else:
            fitness = self.beta * gmean
            
        return fitness
    
    def initialize_harmony_memory(self, X, y):
        """Initialize with tracking"""
        n_features = X.shape[1]
        
        if self.verbose:
            print("\nPhase 1: Initialization")
            print("-" * 50)
        
        # Calculate KL divergence
        self.feature_kl_divergence = self.calculate_feature_kl_divergence(X, y)
        
        # Initialize harmony memory
        self.HM = np.zeros((self.HMS, n_features), dtype=int)
        self.fitness_values = np.zeros(self.HMS)
        
        if self.verbose:
            print("\nInitializing Harmony Memory...")
        
        for j in range(self.HMS):
            for i in range(n_features):
                r = np.random.random()
                p_i = self.sigmoid(self.feature_kl_divergence[i])
                
                if r < p_i:
                    self.HM[j, i] = 1
            
            self.fitness_values[j] = self.calculate_fitness(self.HM[j], X, y)
            
            if self.verbose and (j + 1) % 10 == 0:
                print(f"  Initialized {j + 1}/{self.HMS} harmonies")
        
        # Track best
        best_idx = np.argmax(self.fitness_values)
        self.best_harmony = self.HM[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        
        if self.verbose:
            print(f"\nInitialization complete:")
            print(f"  Best fitness: {self.best_fitness:.4f}")
            print(f"  Selected features: {np.sum(self.best_harmony)}/{n_features}")
    
    def calculate_hamming_distance(self, h1, h2):
        """Calculate Hamming distance between two harmonies"""
        return np.sum(h1 != h2)
    
    def calculate_population_diversity(self):
        """Calculate average Hamming distance in harmony memory"""
        total_distance = 0
        count = 0
        
        for i in range(self.HMS):
            for j in range(i + 1, self.HMS):
                total_distance += self.calculate_hamming_distance(self.HM[i], self.HM[j])
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def improvisation(self, X, y):
        """Improvisation with comprehensive tracking"""
        n_features = X.shape[1]
        
        if self.verbose:
            print("\nPhase 2: Improvisation")
            print("-" * 50)
        
        start_time = time.time()
        
        for iteration in range(self.NI):
            new_harmony = np.zeros(n_features, dtype=int)
            
            for i in range(n_features):
                r1 = np.random.random()
                
                if r1 < self.HMCR:
                    # Memory consideration with KL guidance
                    probs = self.fitness_values * self.feature_kl_divergence[i]
                    if np.sum(probs) > 0:
                        probs = probs / np.sum(probs)
                    else:
                        probs = np.ones(self.HMS) / self.HMS
                    
                    selected_idx = np.random.choice(self.HMS, p=probs)
                    new_harmony[i] = self.HM[selected_idx, i]
                    
                    # Pitch adjustment
                    r2 = np.random.random()
                    if r2 < self.PAR:
                        p_adj = self.sigmoid(self.alpha * self.feature_kl_divergence[i])
                        r3 = np.random.random()
                        if r3 < p_adj:
                            new_harmony[i] = 1 - new_harmony[i]
                else:
                    # Random selection with KL bias
                    r4 = np.random.random()
                    p_i = self.sigmoid(self.feature_kl_divergence[i])
                    new_harmony[i] = 1 if r4 < p_i else 0
            
            # Evaluate new harmony
            new_fitness = self.calculate_fitness(new_harmony, X, y)
            
            # Update harmony memory
            worst_idx = np.argmin(self.fitness_values)
            if new_fitness > self.fitness_values[worst_idx]:
                self.HM[worst_idx] = new_harmony.copy()
                self.fitness_values[worst_idx] = new_fitness
                
                if new_fitness > self.best_fitness:
                    self.best_harmony = new_harmony.copy()
                    self.best_fitness = new_fitness
            
            # Track convergence
            self.fitness_history.append(self.best_fitness)
            selected_features = np.where(self.best_harmony == 1)[0]
            self.kl_history.append(np.sum(self.feature_kl_divergence[selected_features]))
            
            # Calculate feature stability
            if len(self.selected_features_history) > 0:
                prev_features = self.selected_features_history[-1]
                stability = len(set(selected_features) & set(prev_features)) / max(len(selected_features), len(prev_features))
                self.feature_stability.append(stability)
            
            self.selected_features_history.append(selected_features)
            
            # Progress update
            if self.verbose and (iteration + 1) % 25 == 0:
                diversity = self.calculate_population_diversity()
                print(f"  Iteration {iteration + 1}/{self.NI}:")
                print(f"    Best fitness: {self.best_fitness:.4f}")
                print(f"    Selected features: {len(selected_features)}")
                print(f"    Population diversity: {diversity:.3f}")
                print(f"    Total KL: {self.kl_history[-1]:.3f}")
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nImprovisation complete (time: {elapsed_time:.2f}s)")
            print(f"  Final best fitness: {self.best_fitness:.4f}")
            print(f"  Final selected features: {np.sum(self.best_harmony)}/{n_features}")
    
    def fit(self, X, y):
        """Fit with comprehensive tracking"""
        # Validate input
        if len(np.unique(y)) != 2:
            raise ValueError("HKL is designed for binary classification")
        
        # Reset tracking
        self.fitness_history = []
        self.kl_history = []
        self.feature_stability = []
        self.selected_features_history = []
        
        # Run algorithm
        self.initialize_harmony_memory(X, y)
        self.improvisation(X, y)
        
        return self
    
    def transform(self, X):
        """Transform dataset"""
        selected_features = self.get_selected_features()
        return X[:, selected_features]
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self):
        """Get selected feature indices"""
        return np.where(self.best_harmony == 1)[0]
    
    def get_feature_importances(self):
        """Get feature importance scores"""
        return self.feature_kl_divergence
    
    def plot_convergence(self, figsize=(15, 5)):
        """Plot convergence analysis"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Fitness evolution
        axes[0].plot(self.fitness_history, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Best Fitness')
        axes[0].set_title('Fitness Evolution')
        axes[0].grid(True, alpha=0.3)
        
        # KL divergence evolution
        axes[1].plot(self.kl_history, 'g-', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Total KL Divergence')
        axes[1].set_title('KL Divergence Evolution')
        axes[1].grid(True, alpha=0.3)
        
        # Feature stability
        if len(self.feature_stability) > 0:
            axes[2].plot(self.feature_stability, 'r-', linewidth=2)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Feature Stability')
            axes[2].set_title('Feature Selection Stability')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 1.05])
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, top_n=20, figsize=(10, 6)):
        """Plot top feature importances"""
        importances = self.feature_kl_divergence
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=figsize)
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('KL Divergence')
        plt.title(f'Top {top_n} Features by KL Divergence')
        plt.xticks(range(top_n), indices, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def evaluate_performance(self, X, y, cv_folds=5):
        """Comprehensive performance evaluation with cross-validation"""
        selected_features = self.get_selected_features()
        X_selected = X[:, selected_features]
        
        # Initialize metrics storage
        metrics = {
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'gmean': [],
            'auc': []
        }
        
        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in skf.split(X_selected, y):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train classifier
            clf = SVC(kernel='rbf', probability=True, class_weight='balanced', 
                     random_state=self.random_state)
            clf.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sensitivity * specificity)
            auc = roc_auc_score(y_test, y_proba)
            
            metrics['accuracy'].append(accuracy)
            metrics['sensitivity'].append(sensitivity)
            metrics['specificity'].append(specificity)
            metrics['gmean'].append(gmean)
            metrics['auc'].append(auc)
        
        # Calculate mean and std
        results = {}
        for metric, values in metrics.items():
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        return results


# Example usage with comprehensive evaluation
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=200,
        n_informative=30,
        n_redundant=20,
        n_repeated=10,
        n_clusters_per_class=3,
        weights=[0.85, 0.15],  # 85-15 class imbalance
        flip_y=0.02,
        random_state=42
    )
    
    print("Dataset Information:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Imbalance ratio: {np.max(np.bincount(y)) / np.min(np.bincount(y)):.2f}:1")
    
    # Initialize and run HKL
    hkl = HKLExtended(
        HMS=30, 
        HMCR=0.9, 
        PAR=0.3, 
        beta=0.7, 
        NI=100,
        random_state=42,
        verbose=True
    )
    
    # Fit the algorithm
    X_selected = hkl.fit_transform(X, y)
    
    # Get results
    selected_features = hkl.get_selected_features()
    print(f"\nFeature Selection Results:")
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Feature reduction: {(1 - len(selected_features)/X.shape[1])*100:.1f}%")
    
    # Evaluate performance
    print("\nEvaluating performance with 5-fold CV...")
    results = hkl.evaluate_performance(X, y, cv_folds=5)
    
    print("\nPerformance Metrics (mean ± std):")
    for metric in ['accuracy', 'sensitivity', 'specificity', 'gmean', 'auc']:
        mean_val = results[f'{metric}_mean']
        std_val = results[f'{metric}_std']
        print(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Plot convergence
    fig_conv = hkl.plot_convergence()
    plt.savefig('hkl_convergence.png', dpi=300, bbox_inches='tight')
    
    # Plot feature importance
    fig_imp = hkl.plot_feature_importance(top_n=30)
    plt.savefig('hkl_feature_importance.png', dpi=300, bbox_inches='tight')
    
    print("\nPlots saved as 'hkl_convergence.png' and 'hkl_feature_importance.png'")
    
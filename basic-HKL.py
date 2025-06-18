import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


class HKL:
    """
    Harmony search Kullback-Leibler (HKL) algorithm for feature selection
    in high-dimensional imbalanced datasets.
    
    Parameters:
    -----------
    HMS : int, default=30
        Harmony Memory Size
    HMCR : float, default=0.9
        Harmony Memory Consideration Rate
    PAR : float, default=0.3
        Pitch Adjustment Rate
    beta : float, default=0.7
        Weighting parameter for fitness function
    alpha : float, default=1.0
        Scaling parameter for sigmoid function
    NI : int, default=150
        Number of Improvisations (iterations)
    n_bins : int, default=50
        Number of bins for discretization
    """
    
    def __init__(self, HMS=30, HMCR=0.9, PAR=0.3, beta=0.7, alpha=1.0, NI=150, n_bins=50):
        self.HMS = HMS
        self.HMCR = HMCR
        self.PAR = PAR
        self.beta = beta
        self.alpha = alpha
        self.NI = NI
        self.n_bins = n_bins
        self.HM = None
        self.fitness_values = None
        self.feature_kl_divergence = None
        self.best_harmony = None
        self.best_fitness = -np.inf
        self.D_max_KL = 0
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def calculate_kl_divergence_continuous(self, feature_data, y):
        """
        Calculate KL divergence for continuous features using adaptive discretization
        """
        minority_class = np.min(np.bincount(y))
        majority_class = np.max(np.bincount(y))
        
        # Get minority and majority class indices
        minority_idx = np.where(y == np.argmin(np.bincount(y)))[0]
        majority_idx = np.where(y == np.argmax(np.bincount(y)))[0]
        
        # Get feature values for each class
        minority_values = feature_data[minority_idx]
        majority_values = feature_data[majority_idx]
        
        # Adaptive discretization using quantiles
        n_bins = min(self.n_bins, int(np.sqrt(len(feature_data))))
        bins = np.quantile(feature_data, np.linspace(0, 1, n_bins + 1))
        bins[0] = bins[0] - 1e-10  # Ensure all values are included
        bins[-1] = bins[-1] + 1e-10
        
        # Compute histograms
        minority_hist, _ = np.histogram(minority_values, bins=bins)
        majority_hist, _ = np.histogram(majority_values, bins=bins)
        
        # Apply Laplace smoothing
        alpha_smooth = 0.1
        minority_hist = (minority_hist + alpha_smooth) / (len(minority_values) + alpha_smooth * n_bins)
        majority_hist = (majority_hist + alpha_smooth) / (len(majority_values) + alpha_smooth * n_bins)
        
        # Calculate KL divergence
        kl_div = 0
        for i in range(len(minority_hist)):
            if minority_hist[i] > 0:
                kl_div += minority_hist[i] * np.log(minority_hist[i] / majority_hist[i])
                
        return kl_div
    
    def calculate_kl_divergence_categorical(self, feature_data, y):
        """
        Calculate KL divergence for categorical features
        """
        minority_class = np.argmin(np.bincount(y))
        majority_class = np.argmax(np.bincount(y))
        
        # Get minority and majority class indices
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        # Get unique values
        unique_values = np.unique(feature_data)
        
        # Calculate probability mass functions
        minority_pmf = {}
        majority_pmf = {}
        
        for val in unique_values:
            minority_count = np.sum(feature_data[minority_idx] == val)
            majority_count = np.sum(feature_data[majority_idx] == val)
            
            # Apply Laplace smoothing
            alpha_smooth = 0.1
            minority_pmf[val] = (minority_count + alpha_smooth) / (len(minority_idx) + alpha_smooth * len(unique_values))
            majority_pmf[val] = (majority_count + alpha_smooth) / (len(majority_idx) + alpha_smooth * len(unique_values))
        
        # Calculate KL divergence
        kl_div = 0
        for val in unique_values:
            if minority_pmf[val] > 0:
                kl_div += minority_pmf[val] * np.log(minority_pmf[val] / majority_pmf[val])
                
        return kl_div
    
    def calculate_feature_kl_divergence(self, X, y):
        """
        Calculate KL divergence for all features
        """
        n_features = X.shape[1]
        kl_divergences = np.zeros(n_features)
        
        for i in range(n_features):
            feature_data = X[:, i]
            
            # Check if feature is categorical (has few unique values)
            unique_ratio = len(np.unique(feature_data)) / len(feature_data)
            
            if unique_ratio < 0.05:  # Treat as categorical
                kl_divergences[i] = self.calculate_kl_divergence_categorical(feature_data, y)
            else:  # Treat as continuous
                kl_divergences[i] = self.calculate_kl_divergence_continuous(feature_data, y)
                
        return kl_divergences
    
    def calculate_gmean(self, y_true, y_pred):
        """
        Calculate G-mean (geometric mean of sensitivity and specificity)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    
    def evaluate_harmony(self, harmony, X, y):
        """
        Evaluate a harmony vector (feature subset) using SVM classifier
        """
        selected_features = np.where(harmony == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X[:, selected_features], y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train SVM classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm = SVC(kernel='rbf', random_state=42, class_weight='balanced')
        svm.fit(X_train_scaled, y_train)
        
        # Predict and calculate G-mean
        y_pred = svm.predict(X_test_scaled)
        gmean = self.calculate_gmean(y_test, y_pred)
        
        return gmean
    
    def calculate_fitness(self, harmony, X, y):
        """
        Calculate fitness combining G-mean and KL divergence
        """
        # Calculate G-mean
        gmean = self.evaluate_harmony(harmony, X, y)
        
        # Calculate total KL divergence for selected features
        selected_features = np.where(harmony == 1)[0]
        if len(selected_features) > 0:
            total_kl = np.sum(self.feature_kl_divergence[selected_features])
        else:
            total_kl = 0
        
        # Update maximum KL divergence
        if total_kl > self.D_max_KL:
            self.D_max_KL = total_kl
        
        # Calculate fitness
        if self.D_max_KL > 0:
            fitness = self.beta * gmean + (1 - self.beta) * (total_kl / self.D_max_KL)
        else:
            fitness = self.beta * gmean
            
        return fitness
    
    def initialize_harmony_memory(self, X, y):
        """
        Initialize Harmony Memory with KL divergence guidance (Algorithm 2)
        """
        n_features = X.shape[1]
        
        # Calculate KL divergence for each feature
        self.feature_kl_divergence = self.calculate_feature_kl_divergence(X, y)
        
        # Initialize Harmony Memory
        self.HM = np.zeros((self.HMS, n_features), dtype=int)
        self.fitness_values = np.zeros(self.HMS)
        
        for j in range(self.HMS):
            for i in range(n_features):
                # Generate random number
                r = np.random.random()
                
                # Calculate selection probability using sigmoid of KL divergence
                p_i = self.sigmoid(self.feature_kl_divergence[i])
                
                if r < p_i:
                    self.HM[j, i] = 1
                else:
                    self.HM[j, i] = 0
            
            # Evaluate fitness
            self.fitness_values[j] = self.calculate_fitness(self.HM[j], X, y)
        
        # Track best harmony
        best_idx = np.argmax(self.fitness_values)
        self.best_harmony = self.HM[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def improvisation(self, X, y):
        """
        Improvisation process with KL guidance (Algorithm 3)
        """
        n_features = X.shape[1]
        
        for iteration in range(self.NI):
            # Generate new harmony vector
            new_harmony = np.zeros(n_features, dtype=int)
            
            for i in range(n_features):
                r1 = np.random.random()
                
                if r1 < self.HMCR:
                    # Memory consideration with KL guidance
                    # Calculate selection probabilities
                    probs = self.fitness_values * self.feature_kl_divergence[i]
                    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(self.HMS) / self.HMS
                    
                    # Select from harmony memory
                    selected_idx = np.random.choice(self.HMS, p=probs)
                    new_harmony[i] = self.HM[selected_idx, i]
                    
                    # Pitch adjustment
                    r2 = np.random.random()
                    if r2 < self.PAR:
                        # KL-guided pitch adjustment
                        p_adj = self.sigmoid(self.alpha * self.feature_kl_divergence[i])
                        r3 = np.random.random()
                        if r3 < p_adj:
                            new_harmony[i] = 1 - new_harmony[i]  # Flip the bit
                else:
                    # Random selection with KL bias
                    r4 = np.random.random()
                    p_i = self.sigmoid(self.feature_kl_divergence[i])
                    if r4 < p_i:
                        new_harmony[i] = 1
                    else:
                        new_harmony[i] = 0
            
            # Evaluate new harmony
            new_fitness = self.calculate_fitness(new_harmony, X, y)
            
            # Update harmony memory
            worst_idx = np.argmin(self.fitness_values)
            if new_fitness > self.fitness_values[worst_idx]:
                self.HM[worst_idx] = new_harmony.copy()
                self.fitness_values[worst_idx] = new_fitness
                
                # Update best harmony
                if new_fitness > self.best_fitness:
                    self.best_harmony = new_harmony.copy()
                    self.best_fitness = new_fitness
    
    def fit(self, X, y):
        """
        Main HKL algorithm execution
        """
        print("Initializing Harmony Memory...")
        self.initialize_harmony_memory(X, y)
        
        print(f"Initial best fitness: {self.best_fitness:.4f}")
        print(f"Initial selected features: {np.sum(self.best_harmony)}/{X.shape[1]}")
        
        print("\nStarting improvisation process...")
        self.improvisation(X, y)
        
        print(f"\nFinal best fitness: {self.best_fitness:.4f}")
        print(f"Final selected features: {np.sum(self.best_harmony)}/{X.shape[1]}")
        
        return self
    
    def transform(self, X):
        """
        Transform the dataset using selected features
        """
        selected_features = np.where(self.best_harmony == 1)[0]
        return X[:, selected_features]
    
    def fit_transform(self, X, y):
        """
        Fit the algorithm and transform the dataset
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self):
        """
        Get indices of selected features
        """
        return np.where(self.best_harmony == 1)[0]
    
    def get_feature_importances(self):
        """
        Get KL divergence values for all features (as importance scores)
        """
        return self.feature_kl_divergence


# Example usage
if __name__ == "__main__":
    # Generate synthetic imbalanced dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],  # Imbalanced classes
        flip_y=0.02,
        random_state=42
    )
    
    # Initialize and run HKL
    hkl = HKL(HMS=30, HMCR=0.9, PAR=0.3, beta=0.7, NI=100)
    X_selected = hkl.fit_transform(X, y)
    
    # Print results
    selected_features = hkl.get_selected_features()
    print(f"\nSelected features: {selected_features}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Feature reduction: {(1 - len(selected_features)/X.shape[1])*100:.1f}%")
    
    # Evaluate performance on selected features
    from sklearn.metrics import classification_report
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', random_state=42, class_weight='balanced')
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
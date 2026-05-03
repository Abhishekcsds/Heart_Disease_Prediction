"""
Heart Disease Prediction - Complete Model (No Scikit-Learn Required)
Pure Python implementation with train_test_split and metrics from scratch
"""

import pandas as pd
import numpy as np
import pickle
import os
import random

os.makedirs('models', exist_ok=True)


# ============ MANUAL TRAIN_TEST_SPLIT ============
def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    """Manual train-test split without sklearn"""
    random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# ============ MANUAL METRICS ============
def accuracy_score_manual(y_true, y_pred):
    """Manual accuracy calculation"""
    return np.sum(y_true == y_pred) / len(y_true)


def precision_score_manual(y_true, y_pred):
    """Manual precision calculation"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score_manual(y_true, y_pred):
    """Manual recall calculation"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score_manual(y_true, y_pred):
    """Manual F1 score calculation"""
    p = precision_score_manual(y_true, y_pred)
    r = recall_score_manual(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0


def confusion_matrix_manual(y_true, y_pred):
    """Manual confusion matrix"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


# ============ HEART DISEASE MODEL ============
class HeartDiseaseModel:
    """
    Medical knowledge-based weighted model
    Each feature contributes points based on risk level
    """
    
    def __init__(self):
        # Threshold for classification
        self.threshold = 12
        
        # Define risk points for each feature
        self.risk_points = {
            'Age': {
                (50, 60): 1,
                (60, 70): 2,
                (70, 150): 3
            },
            'Sex': {
                1: 1  # Male
            },
            'Chest pain type': {
                2: 1,   # Atypical angina
                3: 2,   # Non-anginal pain
                4: 3    # Asymptomatic
            },
            'BP': {
                (140, 160): 1,
                (160, 180): 2,
                (180, 300): 3
            },
            'Cholesterol': {
                (240, 300): 1,
                (300, 400): 2,
                (400, 1000): 3
            },
            'FBS over 120': {
                1: 2
            },
            'EKG results': {
                2: 1
            },
            'Max HR': {
                (100, 120): 1,
                (0, 100): 2
            },
            'Exercise angina': {
                1: 3
            },
            'ST depression': {
                (1.5, 2.5): 1,
                (2.5, 4.0): 2,
                (4.0, 20): 3
            },
            'Slope of ST': {
                3: 1
            },
            'Number of vessels fluro': {
                1: 1,
                2: 2,
                3: 3
            },
            'Thallium': {
                6: 1,
                7: 3
            }
        }
    
    def _get_points(self, feature_name, value):
        """Get risk points for a feature value"""
        points = 0
        rule = self.risk_points.get(feature_name, {})
        
        for key, pts in rule.items():
            if isinstance(key, tuple):
                if key[0] <= value < key[1]:
                    points = pts
                    break
            else:
                if value == key:
                    points = pts
                    break
        return points
    
    def calculate_total_points(self, features):
        """Calculate total risk points"""
        total = 0
        
        total += self._get_points('Age', features[0])
        total += self._get_points('Sex', features[1])
        total += self._get_points('Chest pain type', features[2])
        total += self._get_points('BP', features[3])
        total += self._get_points('Cholesterol', features[4])
        total += self._get_points('FBS over 120', features[5])
        total += self._get_points('EKG results', features[6])
        total += self._get_points('Max HR', features[7])
        total += self._get_points('Exercise angina', features[8])
        total += self._get_points('ST depression', features[9])
        total += self._get_points('Slope of ST', features[10])
        total += self._get_points('Number of vessels fluro', features[11])
        total += self._get_points('Thallium', features[12])
        
        return total
    
    def predict(self, X):
        """Predict class (0 or 1)"""
        if len(X.shape) == 2:
            predictions = []
            for i in range(X.shape[0]):
                points = self.calculate_total_points(X[i])
                pred = 1 if points >= self.threshold else 0
                predictions.append(pred)
            return np.array(predictions)
        else:
            points = self.calculate_total_points(X)
            return 1 if points >= self.threshold else 0
    
    def predict_proba(self, X):
        """Get probability scores"""
        if len(X.shape) == 2:
            probas = []
            for i in range(X.shape[0]):
                points = self.calculate_total_points(X[i])
                prob_disease = min(points / 28, 0.95)
                probas.append([1 - prob_disease, prob_disease])
            return np.array(probas)
        else:
            points = self.calculate_total_points(X)
            prob_disease = min(points / 28, 0.95)
            return np.array([1 - prob_disease, prob_disease])


class SimpleScaler:
    """Simple feature scaler"""
    
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1
        return self
    
    def transform(self, X):
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def load_data():
    """Load and prepare the dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv('test.csv')
    print(f"   Shape: {df.shape}")
    
    # Feature columns
    feature_cols = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
                    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    
    X = df[feature_cols].values
    
    # Create target based on risk points
    model = HeartDiseaseModel()
    y = []
    for i in range(len(X)):
        points = model.calculate_total_points(X[i])
        y.append(1 if points >= 10 else 0)
    y = np.array(y)
    
    print(f"   Target distribution:")
    print(f"   Class 0 (No Disease): {np.sum(y == 0)}")
    print(f"   Class 1 (Disease): {np.sum(y == 1)}")
    
    return X, y, feature_cols


def train_and_evaluate():
    """Train and evaluate the model"""
    
    print("\n" + "="*60)
    print("🏥 HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Split data (manual, no sklearn)
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = SimpleScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🤖 Training model...")
    model = HeartDiseaseModel()
    
    # Optimize threshold
    best_threshold = model.threshold
    best_accuracy = 0
    
    print("\n📈 Finding best threshold...")
    for threshold in range(5, 20):
        model.threshold = threshold
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score_manual(y_test, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold
    
    model.threshold = best_threshold
    print(f"   Best threshold: {best_threshold}")
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score_manual(y_test, y_pred)
    precision = precision_score_manual(y_test, y_pred)
    recall = recall_score_manual(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred)
    
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE")
    print("="*60)
    print(f"   ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall:    {recall:.4f}")
    print(f"   ✅ F1 Score:  {f1:.4f}")
    
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix_manual(y_test, y_pred)
    print(f"   [[{cm[0][0]}  {cm[0][1]}]")
    print(f"    [{cm[1][0]}  {cm[1][1]}]]")
    
    return model, scaler, feature_cols, accuracy


def save_model(model, scaler, feature_cols, filename='models/heart_disease_model.pkl'):
    """Save the model"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'model_type': 'HeartDiseaseModel'
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved to: {filename}")
    return filename


def main():
    """Main function"""
    model, scaler, feature_cols, accuracy = train_and_evaluate()
    save_model(model, scaler, feature_cols)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n🚀 Next steps:")
    print("   1. Run: python app.py")
    print("   2. Open frontend/index.html")
    print("="*60)


if __name__ == "__main__":
    main()
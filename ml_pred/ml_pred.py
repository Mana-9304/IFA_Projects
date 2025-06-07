import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from collections import defaultdict, deque
import itertools
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SequencePredictor:
    def __init__(self, window_size=10, ensemble_size=3):
        """
        Initialize the sequence prediction system
        
        Args:
            window_size: Number of previous sequences to consider for prediction
            ensemble_size: Number of different models to use in ensemble
        """
        self.window_size = window_size
        self.ensemble_size = ensemble_size
        self.sequences = []
        self.predictions_history = []
        self.accuracy_history = []
        
        # Initialize multiple models for ensemble learning
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'nn': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        self.scalers = {name: StandardScaler() for name in self.models}
        self.model_weights = {name: 1.0 for name in self.models}
        
        # Pattern analysis structures
        self.digit_transitions = defaultdict(lambda: defaultdict(int))
        self.position_patterns = defaultdict(lambda: defaultdict(int))
        self.sequence_patterns = defaultdict(int)
        self.difference_patterns = defaultdict(int)
        
        # Statistical tracking
        self.digit_frequencies = defaultdict(int)
        self.position_frequencies = [defaultdict(int) for _ in range(3)]
        
    def add_sequence(self, sequence: Tuple[int, int, int]):
        """Add a new sequence to the training data"""
        self.sequences.append(sequence)
        self._update_patterns(sequence)
        
    def _update_patterns(self, sequence: Tuple[int, int, int]):
        """Update pattern recognition structures"""
        # Update digit frequencies
        for digit in sequence:
            self.digit_frequencies[digit] += 1
            
        # Update position-specific frequencies
        for pos, digit in enumerate(sequence):
            self.position_frequencies[pos][digit] += 1
            
        # Update transition patterns
        if len(self.sequences) > 1:
            prev_seq = self.sequences[-2]
            for i in range(3):
                self.digit_transitions[prev_seq[i]][sequence[i]] += 1
                
        # Update sequence patterns
        self.sequence_patterns[sequence] += 1
        
        # Update difference patterns
        if len(self.sequences) > 1:
            prev_seq = self.sequences[-2]
            diff = tuple(sequence[i] - prev_seq[i] for i in range(3))
            self.difference_patterns[diff] += 1
    
    def _extract_features(self, sequences: List[Tuple[int, int, int]]) -> np.ndarray:
        """Extract features from sequence history for ML models"""
        if len(sequences) < self.window_size:
            # Pad with zeros if not enough history
            padded = [(0, 0, 0)] * (self.window_size - len(sequences)) + sequences
        else:
            padded = sequences[-self.window_size:]
            
        features = []
        
        # Flatten recent sequences
        for seq in padded:
            features.extend(seq)
            
        # Add statistical features
        recent_sequences = sequences[-min(5, len(sequences)):]
        if recent_sequences:
            # Average of recent sequences
            avg_seq = np.mean(recent_sequences, axis=0)
            features.extend(avg_seq)
            
            # Trend features (differences)
            if len(recent_sequences) > 1:
                trend = np.array(recent_sequences[-1]) - np.array(recent_sequences[-2])
                features.extend(trend)
            else:
                features.extend([0, 0, 0])
                
            # Variance features
            if len(recent_sequences) > 1:
                var_seq = np.var(recent_sequences, axis=0)
                features.extend(var_seq)
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0] * 9)  # padding for statistical features
            
        return np.array(features)
    
    def _train_models(self):
        """Train all models with current sequence data"""
        if len(self.sequences) < self.window_size + 1:
            return
            
        X, y = [], []
        
        # Prepare training data
        for i in range(self.window_size, len(self.sequences)):
            features = self._extract_features(self.sequences[:i])
            target = self.sequences[i]
            X.append(features)
            y.append(target)
            
        if len(X) == 0:
            return
            
        X = np.array(X)
        y = np.array(y)
        
        # Train each model
        for name, model in self.models.items():
            try:
                X_scaled = self.scalers[name].fit_transform(X)
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"Error training {name}: {e}")
    
    def _statistical_prediction(self) -> List[Tuple[int, int, int]]:
        """Generate predictions based on statistical patterns"""
        predictions = []
        
        if not self.sequences:
            return [(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)) 
                   for _ in range(20)]
        
        last_seq = self.sequences[-1]
        
        # Method 1: Most frequent transitions
        for _ in range(7):
            pred = []
            for i in range(3):
                transitions = self.digit_transitions[last_seq[i]]
                if transitions:
                    next_digit = max(transitions.items(), key=lambda x: x[1])[0]
                else:
                    # Fallback to most frequent digit in this position
                    pos_freq = self.position_frequencies[i]
                    if pos_freq:
                        next_digit = max(pos_freq.items(), key=lambda x: x[1])[0]
                    else:
                        next_digit = np.random.randint(0, 10)
                pred.append(next_digit)
            predictions.append(tuple(pred))
        
        # Method 2: Difference pattern continuation
        if len(self.sequences) > 1:
            for _ in range(6):
                prev_seq = self.sequences[-2]
                diff = tuple(last_seq[i] - prev_seq[i] for i in range(3))
                pred = tuple((last_seq[i] + diff[i]) % 10 for i in range(3))
                predictions.append(pred)
        
        # Method 3: Most frequent sequences with variation
        for _ in range(4):
            if self.sequence_patterns:
                base_seq = max(self.sequence_patterns.items(), key=lambda x: x[1])[0]
                # Add small random variation
                pred = tuple((base_seq[i] + np.random.randint(-2, 3)) % 10 for i in range(3))
                predictions.append(pred)
            else:
                pred = tuple(np.random.randint(0, 10) for _ in range(3))
                predictions.append(pred)
        
        # Method 4: Random based on position frequencies
        for _ in range(3):
            pred = []
            for i in range(3):
                pos_freq = self.position_frequencies[i]
                if pos_freq:
                    # Weighted random selection
                    digits, weights = zip(*pos_freq.items())
                    digit = np.random.choice(digits, p=np.array(weights)/sum(weights))
                else:
                    digit = np.random.randint(0, 10)
                pred.append(digit)
            predictions.append(tuple(pred))
        
        return predictions[:20]
    
    def _ml_prediction(self) -> List[Tuple[int, int, int]]:
        """Generate predictions using ML models"""
        predictions = []
        
        if len(self.sequences) < self.window_size + 1:
            return self._statistical_prediction()
        
        try:
            current_features = self._extract_features(self.sequences).reshape(1, -1)
            
            model_predictions = {}
            for name, model in self.models.items():
                try:
                    features_scaled = self.scalers[name].transform(current_features)
                    pred = model.predict(features_scaled)[0]
                    # Round and clip to valid range
                    pred = tuple(max(0, min(9, int(round(p)))) for p in pred)
                    model_predictions[name] = pred
                except Exception as e:
                    print(f"Error in {name} prediction: {e}")
                    model_predictions[name] = tuple(np.random.randint(0, 10) for _ in range(3))
            
            # Weighted ensemble predictions
            for _ in range(10):
                ensemble_pred = [0, 0, 0]
                total_weight = sum(self.model_weights.values())
                
                for name, pred in model_predictions.items():
                    weight = self.model_weights[name] / total_weight
                    for i in range(3):
                        ensemble_pred[i] += pred[i] * weight
                
                # Add some randomness and round
                final_pred = tuple(max(0, min(9, int(round(p + np.random.normal(0, 0.5))))) 
                                 for p in ensemble_pred)
                predictions.append(final_pred)
            
            # Add individual model predictions
            for pred in model_predictions.values():
                predictions.append(pred)
                if len(predictions) >= 20:
                    break
                    
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._statistical_prediction()
        
        # Fill remaining with statistical predictions if needed
        while len(predictions) < 20:
            predictions.extend(self._statistical_prediction())
        
        return predictions[:20]
    
    def predict_next_sequences(self, num_predictions=20) -> List[Tuple[int, int, int]]:
        """Generate predictions for the next sequence"""
        if len(self.sequences) < 3:
            # Not enough data, use random predictions
            return [(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)) 
                   for _ in range(num_predictions)]
        
        # Train models with current data
        self._train_models()
        
        # Combine ML and statistical predictions
        ml_preds = self._ml_prediction()
        stat_preds = self._statistical_prediction()
        
        # Merge and deduplicate
        all_preds = ml_preds + stat_preds
        seen = set()
        unique_preds = []
        
        for pred in all_preds:
            if pred not in seen:
                unique_preds.append(pred)
                seen.add(pred)
            if len(unique_preds) >= num_predictions:
                break
        
        # Fill with random if needed
        while len(unique_preds) < num_predictions:
            pred = tuple(np.random.randint(0, 10) for _ in range(3))
            if pred not in seen:
                unique_preds.append(pred)
                seen.add(pred)
        
        return unique_preds[:num_predictions]
    
    def update_with_feedback(self, predictions: List[Tuple[int, int, int]], 
                           actual: Tuple[int, int, int]):
        """Update model performance based on prediction accuracy"""
        self.predictions_history.append(predictions)
        
        # Calculate accuracy metrics
        accuracies = []
        for pred in predictions:
            # Digit-wise accuracy
            digit_acc = sum(1 for i in range(3) if pred[i] == actual[i]) / 3
            accuracies.append(digit_acc)
        
        best_accuracy = max(accuracies)
        avg_accuracy = np.mean(accuracies)
        
        self.accuracy_history.append({
            'best': best_accuracy,
            'average': avg_accuracy,
            'predictions': predictions,
            'actual': actual
        })
        
        # Update model weights based on performance
        if len(self.sequences) > self.window_size:
            try:
                # Test each model's prediction accuracy
                features = self._extract_features(self.sequences[:-1]).reshape(1, -1)
                
                for name, model in self.models.items():
                    try:
                        features_scaled = self.scalers[name].transform(features)
                        pred = model.predict(features_scaled)[0]
                        pred_rounded = tuple(max(0, min(9, int(round(p)))) for p in pred)
                        
                        # Calculate model-specific accuracy
                        model_acc = sum(1 for i in range(3) if pred_rounded[i] == actual[i]) / 3
                        
                        # Update weight based on accuracy
                        self.model_weights[name] = 0.7 * self.model_weights[name] + 0.3 * (model_acc + 0.1)
                        
                    except Exception as e:
                        print(f"Error updating weight for {name}: {e}")
                        
            except Exception as e:
                print(f"Error in feedback update: {e}")
        
        print(f"Prediction Feedback:")
        print(f"  Actual sequence: {actual}")
        print(f"  Best prediction accuracy: {best_accuracy:.2%}")
        print(f"  Average prediction accuracy: {avg_accuracy:.2%}")
        print(f"  Model weights: {dict(self.model_weights)}")
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of patterns and performance"""
        if not self.sequences:
            return {"error": "No sequences available for analysis"}
        
        analysis = {
            "sequence_count": len(self.sequences),
            "recent_sequences": self.sequences[-10:],
            "digit_frequencies": dict(self.digit_frequencies),
            "position_patterns": [dict(pos_freq) for pos_freq in self.position_frequencies],
            "most_common_transitions": {},
            "most_common_differences": dict(sorted(self.difference_patterns.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10]),
            "model_weights": dict(self.model_weights),
            "prediction_accuracy_trend": self.accuracy_history[-10:] if self.accuracy_history else []
        }
        
        # Analyze transitions
        for digit in range(10):
            if digit in self.digit_transitions:
                transitions = self.digit_transitions[digit]
                if transitions:
                    most_common = max(transitions.items(), key=lambda x: x[1])
                    analysis["most_common_transitions"][digit] = most_common
        
        return analysis

# Example usage and testing
def main():
    """Main function to demonstrate the sequence predictor"""
    predictor = SequencePredictor()
    
    # Sample initial sequences (at least 50 as requested)
    initial_sequences = [
        (0, 1, 2), (3, 4, 9), (5, 6, 7), (2, 8, 1), (9, 0, 3),
        (4, 7, 5), (1, 9, 8), (6, 2, 4), (8, 5, 0), (3, 1, 7),
        (7, 4, 6), (0, 8, 9), (5, 3, 2), (9, 6, 1), (2, 7, 4),
        (4, 0, 8), (1, 5, 3), (8, 9, 7), (6, 2, 0), (3, 4, 5),
        (7, 1, 9), (0, 6, 2), (5, 8, 4), (9, 3, 1), (2, 5, 7),
        (4, 9, 0), (1, 7, 6), (8, 2, 3), (6, 0, 5), (3, 8, 9),
        (7, 5, 1), (0, 9, 4), (5, 2, 8), (9, 7, 0), (2, 4, 6),
        (4, 1, 3), (1, 6, 9), (8, 0, 5), (6, 7, 2), (3, 9, 4),
        (7, 2, 8), (0, 5, 1), (5, 9, 6), (9, 4, 3), (2, 8, 0),
        (4, 6, 7), (1, 3, 5), (8, 7, 9), (6, 1, 4), (3, 5, 2),
        (7, 8, 0), (0, 4, 6), (5, 1, 9), (9, 2, 7), (2, 6, 3)
    ]
    
    print("=== Sequence Prediction Algorithm Demo ===\n")
    
    # Add initial sequences
    print("Adding initial sequences...")
    for seq in initial_sequences:
        predictor.add_sequence(seq)
    
    print(f"Added {len(initial_sequences)} initial sequences")
    print(f"Starting with last sequence: {initial_sequences[-1]}")
    
    # Simulation loop
    for round_num in range(5):
        print(f"\n--- Prediction Round {round_num + 1} ---")
        
        # Make predictions
        predictions = predictor.predict_next_sequences(20)
        print(f"Generated {len(predictions)} predictions:")
        for i, pred in enumerate(predictions[:10], 1):  # Show first 10
            print(f"  {i:2d}. {pred}")
        if len(predictions) > 10:
            print(f"  ... and {len(predictions) - 10} more")
        
        # Simulate actual next sequence (in real usage, this would come from your data source)
        actual_next = (np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10))
        print(f"\nActual next sequence: {actual_next}")
        
        # Add the actual sequence and update with feedback
        predictor.add_sequence(actual_next)
        predictor.update_with_feedback(predictions, actual_next)
    
    # Show final analysis
    print("\n=== Final Analysis ===")
    analysis = predictor.get_analysis()
    
    print(f"Total sequences processed: {analysis['sequence_count']}")
    print(f"Recent sequences: {analysis['recent_sequences']}")
    print(f"Most common digit transitions: {analysis['most_common_transitions']}")
    print(f"Model weights: {analysis['model_weights']}")
    
    if analysis['prediction_accuracy_trend']:
        avg_best = np.mean([acc['best'] for acc in analysis['prediction_accuracy_trend']])
        avg_overall = np.mean([acc['average'] for acc in analysis['prediction_accuracy_trend']])
        print(f"Average best prediction accuracy: {avg_best:.2%}")
        print(f"Average overall prediction accuracy: {avg_overall:.2%}")

if __name__ == "__main__":
    main()
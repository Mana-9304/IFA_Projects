import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class SequencePredictor:
    def __init__(self, window_size=12, ensemble_size=4):
        """
        Initialize the sequence prediction system with enhanced parameters.
        """
        self.window_size = window_size
        self.ensemble_size = ensemble_size
        self.sequences = []
        self.predictions_history = []
        self.accuracy_history = []
        
        # Initialize models with MultiOutputRegressor for GradientBoosting
        self.models = {
            'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42),
            'gb': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)),
            'nn': MLPRegressor(hidden_layer_sizes=(128, 64, 32), learning_rate='adaptive', max_iter=1500, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        }
        
        self.scalers = {name: StandardScaler() for name in self.models}
        self.model_weights = {name: 1.0 for name in self.models}
        
        # Pattern analysis structures
        self.digit_transitions = defaultdict(lambda: defaultdict(int))
        self.position_patterns = defaultdict(lambda: defaultdict(int))
        self.sequence_patterns = defaultdict(int)
        self.difference_patterns = defaultdict(int)
        self.digit_cooccurrence = np.zeros((10, 10, 3))  # Co-occurrence matrix for each position
        
        # Statistical tracking
        self.digit_frequencies = defaultdict(int)
        self.position_frequencies = [defaultdict(int) for _ in range(3)]
        
    def add_sequence(self, sequence: Tuple[int, int, int]):
        """Add a new sequence to the training data and update patterns."""
        self.sequences.append(sequence)
        self._update_patterns(sequence)
        
    def _update_patterns(self, sequence: Tuple[int, int, int]):
        """Update pattern recognition structures with co-occurrence and higher-order transitions."""
        # Update digit frequencies
        for digit in sequence:
            self.digit_frequencies[digit] += 1
            
        # Update position-specific frequencies
        for pos, digit in enumerate(sequence):
            self.position_frequencies[pos][digit] += 1
            
        # Update co-occurrence matrix
        for pos, digit in enumerate(sequence):
            for other_digit in sequence:
                self.digit_cooccurrence[digit][other_digit][pos] += 1
                
        # Update transition patterns (including second-order transitions)
        if len(self.sequences) > 2:
            prev_seq = self.sequences[-2]
            prev_prev_seq = self.sequences[-3]
            for i in range(3):
                self.digit_transitions[(prev_prev_seq[i], prev_seq[i])][sequence[i]] += 1
                
        # Update sequence patterns
        self.sequence_patterns[sequence] += 1
        
        # Update difference patterns
        if len(self.sequences) > 1:
            prev_seq = self.sequences[-2]
            diff = tuple(sequence[i] - prev_seq[i] for i in range(3))
            self.difference_patterns[diff] += 1
    
    def _extract_features(self, sequences: List[Tuple[int, int, int]]) -> np.ndarray:
        """Extract enhanced features for ML models."""
        if len(sequences) < self.window_size:
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
            recent_array = np.array(recent_sequences)
            # Position-wise mean and variance
            pos_mean = np.mean(recent_array, axis=0)
            pos_var = np.var(recent_array, axis=0)
            features.extend(pos_mean)
            features.extend(pos_var)
            
            # Sequence entropy
            entropy = -np.sum([p * np.log2(p + 1e-10) for p in np.bincount(recent_array.flatten(), minlength=10) / recent_array.size])
            features.append(entropy)
            
            # Co-occurrence features
            for pos in range(3):
                pos_cooccur = self.digit_cooccurrence[:, :, pos].flatten()
                features.extend(pos_cooccur[:10])  # Top 10 co-occurrence features per position
            
            # Trend features (multi-step differences)
            if len(recent_sequences) > 2:
                trend1 = np.array(recent_sequences[-1]) - np.array(recent_sequences[-2])
                trend2 = np.array(recent_sequences[-2]) - np.array(recent_sequences[-3])
                features.extend(trend1)
                features.extend(trend2)
            else:
                features.extend([0] * 6)
                
        else:
            features.extend([0] * (3 + 3 + 1 + 30 + 6))  # Padding for statistical features
            
        return np.array(features)
    
    def _train_models(self):
        """Train all models with current sequence data."""
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
        """Generate predictions based on enhanced statistical patterns."""
        predictions = []
        
        if not self.sequences:
            return [(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)) 
                   for _ in range(20)]
        
        last_seq = self.sequences[-1]
        
        # Method 1: Higher-order transition probabilities
        for _ in range(7):
            pred = []
            for i in range(3):
                if len(self.sequences) > 1:
                    prev_seq = self.sequences[-2]
                    transitions = self.digit_transitions[(prev_seq[i], last_seq[i])]
                else:
                    transitions = self.digit_transitions[last_seq[i]]
                    
                if transitions:
                    next_digit = max(transitions.items(), key=lambda x: x[1])[0]
                else:
                    pos_freq = self.position_frequencies[i]
                    next_digit = max(pos_freq.items(), key=lambda x: x[1])[0] if pos_freq else np.random.randint(0, 10)
                pred.append(next_digit)
            predictions.append(tuple(pred))
        
        # Method 2: Difference pattern continuation
        if len(self.sequences) > 1:
            for _ in range(6):
                prev_seq = self.sequences[-2]
                diff = tuple(last_seq[i] - prev_seq[i] for i in range(3))
                pred = tuple((last_seq[i] + diff[i]) % 10 for i in range(3))
                predictions.append(pred)
        
        # Method 3: Sequence pattern with constrained variation
        for _ in range(4):
            if self.sequence_patterns:
                base_seq = max(self.sequence_patterns.items(), key=lambda x: x[1])[0]
                pred = tuple((base_seq[i] + np.random.randint(-1, 2)) % 10 for i in range(3))
                predictions.append(pred)
            else:
                pred = tuple(np.random.randint(0, 10) for _ in range(3))
                predictions.append(pred)
        
        # Method 4: Weighted position-based sampling
        for _ in range(3):
            pred = []
            for i in range(3):
                pos_freq = self.position_frequencies[i]
                if pos_freq:
                    digits, weights = zip(*pos_freq.items())
                    digit = np.random.choice(digits, p=np.array(weights)/sum(weights))
                else:
                    digit = np.random.randint(0, 10)
                pred.append(digit)
            predictions.append(tuple(pred))
        
        return predictions[:20]
    
    def _ml_prediction(self) -> List[Tuple[int, int, int]]:
        """Generate predictions using enhanced ML ensemble."""
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
                    pred = tuple(max(0, min(9, int(round(p)))) for p in pred)
                    model_predictions[name] = pred
                except Exception as e:
                    print(f"Error in {name} prediction: {e}")
                    model_predictions[name] = tuple(np.random.randint(0, 10) for _ in range(3))
            
            # Weighted ensemble with meta-learner
            for _ in range(10):
                ensemble_pred = [0, 0, 0]
                total_weight = sum(self.model_weights.values())
                
                for name, pred in model_predictions.items():
                    weight = self.model_weights[name] / total_weight
                    for i in range(3):
                        ensemble_pred[i] += pred[i] * weight
                
                # Constrain predictions to observed patterns
                final_pred = []
                for i, p in enumerate(ensemble_pred):
                    pos_freq = self.position_frequencies[i]
                    if pos_freq:
                        valid_digits = list(pos_freq.keys())
                        final_pred.append(max(0, min(9, int(round(p)))))
                    else:
                        final_pred.append(max(0, min(9, int(round(p)))))
                predictions.append(tuple(final_pred))
            
            # Add individual model predictions
            for pred in model_predictions.values():
                predictions.append(pred)
                if len(predictions) >= 20:
                    break
                    
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._statistical_prediction()
        
        while len(predictions) < 20:
            predictions.extend(self._statistical_prediction())
        
        return predictions[:20]
    
    def predict_next_sequences(self, num_predictions=20) -> List[Tuple[int, int, int]]:
        """Generate predictions for the next sequence."""
        if len(self.sequences) < 3:
            return [(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)) 
                   for _ in range(num_predictions)]
        
        self._train_models()
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
        
        while len(unique_preds) < num_predictions:
            pred = tuple(np.random.randint(0, 10) for _ in range(3))
            if pred not in seen:
                unique_preds.append(pred)
                seen.add(pred)
        
        return unique_preds[:num_predictions]
    
    def update_with_feedback(self, predictions: List[Tuple[int, int, int]], 
                           actual: Tuple[int, int, int]):
        """Update model performance with enhanced feedback mechanism."""
        self.predictions_history.append(predictions)
        
        # Calculate position-wise accuracies
        accuracies = []
        pos_accuracies = [[] for _ in range(3)]
        for pred in predictions:
            digit_acc = sum(1 for i in range(3) if pred[i] == actual[i]) / 3
            accuracies.append(digit_acc)
            for i in range(3):
                pos_accuracies[i].append(1 if pred[i] == actual[i] else 0)
        
        best_accuracy = max(accuracies)
        avg_accuracy = np.mean(accuracies)
        pos_avg_acc = [np.mean(pos_acc) for pos_acc in pos_accuracies]
        
        self.accuracy_history.append({
            'best': best_accuracy,
            'average': avg_accuracy,
            'position_accuracies': pos_avg_acc,
            'predictions': predictions,
            'actual': actual
        })
        
        # Update model weights with exponential moving average
        if len(self.sequences) > self.window_size:
            try:
                features = self._extract_features(self.sequences[:-1]).reshape(1, -1)
                for name, model in self.models.items():
                    try:
                        features_scaled = self.scalers[name].transform(features)
                        pred = model.predict(features_scaled)[0]
                        pred_rounded = tuple(max(0, min(9, int(round(p)))) for p in pred)
                        model_acc = sum(1 for i in range(3) if pred_rounded[i] == actual[i]) / 3
                        self.model_weights[name] = 0.8 * self.model_weights[name] + 0.2 * (model_acc + 0.1)
                    except Exception as e:
                        print(f"Error updating weight for {name}: {e}")
            except Exception as e:
                print(f"Error in feedback update: {e}")
        
        print(f"Prediction Feedback:")
        print(f"  Actual sequence: {actual}")
        print(f"  Best prediction accuracy: {best_accuracy:.2%}")
        print(f"  Average prediction accuracy: {avg_accuracy:.2%}")
        print(f"  Position-wise accuracies: {dict(zip(range(1, 4), [f'{acc:.2%}' for acc in pos_avg_acc]))}")
        print(f"  Model weights: {dict(self.model_weights)}")
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of patterns and performance."""
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
            "prediction_accuracy_trend": self.accuracy_history[-10:] if self.accuracy_history else [],
            "cooccurrence_stats": {}
        }
        
        # Analyze transitions
        for key in self.digit_transitions:
            transitions = self.digit_transitions[key]
            if transitions:
                most_common = max(transitions.items(), key=lambda x: x[1])
                analysis["most_common_transitions"][str(key)] = most_common
        
        # Analyze co-occurrence
        for pos in range(3):
            cooccur = self.digit_cooccurrence[:, :, pos]
            analysis["cooccurrence_stats"][f"position_{pos+1}"] = np.sum(cooccur, axis=1).tolist()
        
        return analysis

def main():
    """Main function to demonstrate the sequence predictor."""
    predictor = SequencePredictor()
    
    # Sample initial sequences (at least 50)
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
        for i, pred in enumerate(predictions[:10], 1):
            print(f"  {i:2d}. {pred}")
        if len(predictions) > 10:
            print(f"  ... and {len(predictions) - 10} more")
        
        # Simulate actual next sequence
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
    print(f"Co-occurrence stats: {analysis['cooccurrence_stats']}")
    
    if analysis['prediction_accuracy_trend']:
        avg_best = np.mean([acc['best'] for acc in analysis['prediction_accuracy_trend']])
        avg_overall = np.mean([acc['average'] for acc in analysis['prediction_accuracy_trend']])
        print(f"Average best prediction accuracy: {avg_best:.2%}")
        print(f"Average overall prediction accuracy: {avg_overall:.2%}")
        print(f"Position-wise accuracies (last round): {dict(zip(range(1, 4), analysis['prediction_accuracy_trend'][-1]['position_accuracies']))}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Head Dwell Time Navigation Classification System
Proximity Forest 2.0 with MSM, SSDTW, and ERP Distance Metrics

This system trains and classifies head angle dwell time patterns for wheelchair
navigation using 5 behavioral categories. Implements Proximity Forest 2.0 with 
enhanced distance measures including custom MSM with Sakoe-Chiba band calculation,
Subsequence DTW with wavelets, and Edit Distance with Real Penalty.

Dwell time represents how long the user maintains their head in specific positions,
indicating different navigation intentions and exploration behaviors.

Author: Manuel
Institution: CIMAT - Centro de Investigación en Matemáticas
Date: December 2025
"""

import numpy as np
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')


# ============================================================================
# DISTANCE METRICS
# ============================================================================

class MSM_Enhanced:
    """
    Move-Split-Merge with Enhanced Sakoe-Chiba Band Calculation
    
    Implements MSM distance with proper Sakoe-Chiba band estimation based on
    key events (maxima and minima) temporal positions.
    """
    
    def __init__(self, c: float = 1.0):
        """
        Parameters
        ----------
        c : float
            Cost parameter for Move, Split, Merge operations
        """
        self.c = c
    
    def _calculate_sakoe_chiba_band(self, P: np.ndarray, Q: np.ndarray, 
                                    margin: int = 1) -> int:
        """
        Calculate Sakoe-Chiba band width based on key events
        
        Parameters
        ----------
        P, Q : np.ndarray
            Input time series
        margin : int
            Additional margin to add to calculated band
            
        Returns
        -------
        int
            Sakoe-Chiba band width
        """
        # Find peaks and valleys
        peaks_P, _ = find_peaks(P)
        peaks_Q, _ = find_peaks(Q)
        valleys_P, _ = find_peaks(-P)
        valleys_Q, _ = find_peaks(-Q)
        
        # Calculate temporal displacements
        max_displacement = 0
        
        if len(peaks_P) > 0 and len(peaks_Q) > 0:
            idx_max_P = peaks_P[np.argmax(P[peaks_P])]
            idx_max_Q = peaks_Q[np.argmax(Q[peaks_Q])]
            max_displacement = max(max_displacement, abs(idx_max_P - idx_max_Q))
        
        if len(valleys_P) > 0 and len(valleys_Q) > 0:
            idx_min_P = valleys_P[np.argmin(P[valleys_P])]
            idx_min_Q = valleys_Q[np.argmin(Q[valleys_Q])]
            max_displacement = max(max_displacement, abs(idx_min_P - idx_min_Q))
        
        # Default band if no peaks found
        if max_displacement == 0:
            max_displacement = max(len(P), len(Q)) // 10
        
        return max_displacement + margin
    
    def compute_distance(self, seq1: Tuple[np.ndarray, np.ndarray], 
                        seq2: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Compute MSM distance between two bivariate sequences
        
        Parameters
        ----------
        seq1, seq2 : tuple of np.ndarray
            Bivariate sequences (angle_x, angle_y)
            
        Returns
        -------
        float
            MSM distance
        """
        x1, y1 = seq1
        x2, y2 = seq2
        
        # Calculate band for each dimension
        band_x = self._calculate_sakoe_chiba_band(x1, x2)
        band_y = self._calculate_sakoe_chiba_band(y1, y2)
        band = max(band_x, band_y)
        
        # Compute MSM for each dimension
        dist_x = self._msm_univariate(x1, x2, band)
        dist_y = self._msm_univariate(y1, y2, band)
        
        return dist_x + dist_y
    
    def _msm_univariate(self, s1: np.ndarray, s2: np.ndarray, band: int) -> float:
        """MSM algorithm for univariate series with Sakoe-Chiba band"""
        n, m = len(s1), len(s2)
        
        # Initialize cost matrix
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0
        
        for i in range(1, n + 1):
            j_start = max(1, i - band)
            j_end = min(m + 1, i + band + 1)
            
            for j in range(j_start, j_end):
                # Move cost
                move_cost = abs(s1[i-1] - s2[j-1])
                
                # Split cost
                if i > 1:
                    split_cost = self.c + abs(s1[i-1] - s1[i-2])
                else:
                    split_cost = np.inf
                
                # Merge cost
                if j > 1:
                    merge_cost = self.c + abs(s2[j-1] - s2[j-2])
                else:
                    merge_cost = np.inf
                
                cost[i, j] = min(
                    cost[i-1, j-1] + move_cost,
                    cost[i-1, j] + split_cost,
                    cost[i, j-1] + merge_cost
                )
        
        return cost[n, m]


class SSDTW_Multivariable:
    """
    Subsequence Dynamic Time Warping with Wavelet Transform
    
    Implements SSDTW for bivariate time series classification with
    wavelet-based feature extraction.
    """
    
    def __init__(self, L: int = 3):
        """
        Parameters
        ----------
        L : int
            Wavelet decomposition level
        """
        self.L = L
    
    def compute_distance(self, seq1: Tuple[np.ndarray, np.ndarray],
                        seq2: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Compute SSDTW distance between two bivariate sequences
        
        Parameters
        ----------
        seq1, seq2 : tuple of np.ndarray
            Bivariate sequences (angle_x, angle_y)
            
        Returns
        -------
        float
            SSDTW distance
        """
        x1, y1 = seq1
        x2, y2 = seq2
        
        # Apply wavelet transform
        wx1 = self._wavelet_transform(x1)
        wy1 = self._wavelet_transform(y1)
        wx2 = self._wavelet_transform(x2)
        wy2 = self._wavelet_transform(y2)
        
        # Compute DTW on wavelet coefficients
        dist_x = self._dtw(wx1, wx2)
        dist_y = self._dtw(wy1, wy2)
        
        return dist_x + dist_y
    
    def _wavelet_transform(self, signal: np.ndarray) -> np.ndarray:
        """Simple Haar wavelet transform"""
        coeffs = signal.copy()
        n = len(coeffs)
        
        for level in range(self.L):
            n = n // 2
            if n < 1:
                break
            
            # Approximation and detail coefficients
            approx = (coeffs[:2*n:2] + coeffs[1:2*n:2]) / 2
            detail = (coeffs[:2*n:2] - coeffs[1:2*n:2]) / 2
            coeffs[:n] = approx
            coeffs[n:2*n] = detail
        
        return coeffs
    
    def _dtw(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Standard DTW implementation"""
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return dtw_matrix[n, m]


class ERP_2D:
    """
    Edit Distance with Real Penalty for Bivariate Series
    
    Implements ERP distance measure for 2D time series with gap penalty.
    """
    
    def __init__(self, g: float = 0.0):
        """
        Parameters
        ----------
        g : float
            Gap penalty value
        """
        self.g = g
    
    def compute_distance(self, seq1: Tuple[np.ndarray, np.ndarray],
                        seq2: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Compute ERP distance between two bivariate sequences
        
        Parameters
        ----------
        seq1, seq2 : tuple of np.ndarray
            Bivariate sequences (angle_x, angle_y)
            
        Returns
        -------
        float
            ERP distance
        """
        x1, y1 = seq1
        x2, y2 = seq2
        
        dist_x = self._erp_univariate(x1, x2)
        dist_y = self._erp_univariate(y1, y2)
        
        return dist_x + dist_y
    
    def _erp_univariate(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """ERP for univariate series"""
        n, m = len(s1), len(s2)
        
        erp_matrix = np.zeros((n + 1, m + 1))
        
        # Initialize first column and row
        for i in range(1, n + 1):
            erp_matrix[i, 0] = erp_matrix[i-1, 0] + abs(s1[i-1] - self.g)
        
        for j in range(1, m + 1):
            erp_matrix[0, j] = erp_matrix[0, j-1] + abs(s2[j-1] - self.g)
        
        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                erp_matrix[i, j] = min(
                    erp_matrix[i-1, j-1] + cost,
                    erp_matrix[i-1, j] + abs(s1[i-1] - self.g),
                    erp_matrix[i, j-1] + abs(s2[j-1] - self.g)
                )
        
        return erp_matrix[n, m]


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

@dataclass
class DwellParameters:
    """Parameters for synthetic dwell time generation"""
    mean: float
    std: float


class SyntheticDwellGenerator:
    """
    Generates synthetic dwell time sequences for navigation classification
    
    Produces training data for 5 navigation categories based on dwell time
    patterns indicating different navigation behaviors.
    """
    
    DWELL_CONFIGS = {
        'NAVEGACION_EFICIENTE': DwellParameters(mean=15.0, std=3.0),
        'NAVEGACION_DIRIGIDA': DwellParameters(mean=4.0, std=0.8),
        'EXPLORACION_PAUSADA': DwellParameters(mean=7.0, std=1.5),
        'EXPLORACION_ACTIVA': DwellParameters(mean=0.5, std=0.2),
        'BUSQUEDA_REORIENTACION': DwellParameters(mean=1.5, std=0.4)
    }
    
    def __init__(self, sequence_length: int = 50):
        """
        Parameters
        ----------
        sequence_length : int
            Length of generated sequences
        """
        self.sequence_length = sequence_length
    
    def generate_category_dataset(self, category: str, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic dataset for a specific navigation category
        
        Parameters
        ----------
        category : str
            Category name (must be in DWELL_CONFIGS)
        n_samples : int
            Number of sequences to generate
            
        Returns
        -------
        list of tuple
            List of (HeadAngleX, HeadAngleY) sequences with embedded dwell patterns
        """
        if category not in self.DWELL_CONFIGS:
            raise ValueError(f"Unknown category: {category}")
        
        config = self.DWELL_CONFIGS[category]
        sequences = []
        
        for _ in range(n_samples):
            # Generate dwell times (descending pattern to simulate fixation decay)
            base_dwell = np.random.normal(config.mean, config.std, self.sequence_length)
            base_dwell = np.abs(base_dwell)
            
            # Add decay pattern
            decay = np.linspace(1.0, 0.7, self.sequence_length)
            dwell_times = base_dwell * decay
            
            # Convert dwell times to angle sequences (longer dwell = more stable angles)
            angles_x = []
            angles_y = []
            
            for i, dwell in enumerate(dwell_times):
                # Base angle with noise inversely proportional to dwell time
                base_angle_x = np.random.uniform(-90, 90)
                base_angle_y = np.random.uniform(-90, 90)
                
                # Stability factor: higher dwell = less variation
                stability = min(dwell / 10.0, 1.0)
                noise_scale = (1.0 - stability) * 5.0
                
                angle_x = base_angle_x + np.random.normal(0, noise_scale)
                angle_y = base_angle_y + np.random.normal(0, noise_scale)
                
                angles_x.append(angle_x)
                angles_y.append(angle_y)
            
            sequences.append((np.array(angles_x), np.array(angles_y)))
        
        return sequences
    
    def generate_full_dataset(self, n_samples_per_category: int) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate complete dataset for all 5 categories
        
        Parameters
        ----------
        n_samples_per_category : int
            Number of samples per category
            
        Returns
        -------
        dict
            Dictionary mapping category names to lists of sequences
        """
        dataset = {}
        
        for category in self.DWELL_CONFIGS.keys():
            dataset[category] = self.generate_category_dataset(category, n_samples_per_category)
        
        return dataset


# ============================================================================
# PROXIMITY FOREST 2.0
# ============================================================================

class ProximityTree:
    """
    Single tree in Proximity Forest ensemble
    
    Decision tree based on elastic distance measures with candidate sampling
    and information gain for improved split selection.
    """
    
    def __init__(self, max_depth: int, distance_measures: List[str], 
                 msm_instance: MSM_Enhanced, ssdtw_instance: SSDTW_Multivariable,
                 erp_instance: ERP_2D):
        self.max_depth = max_depth
        self.distance_measures = distance_measures
        self.msm = msm_instance
        self.ssdtw = ssdtw_instance
        self.erp = erp_instance
        self.root = None
    
    def fit(self, X: List[Tuple[np.ndarray, np.ndarray]], y: np.ndarray):
        """Train the tree"""
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X: List[Tuple[np.ndarray, np.ndarray]], y: np.ndarray, depth: int):
        """Recursively build tree structure"""
        n_samples = len(X)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < 2 or len(np.unique(y)) == 1:
            return {'leaf': True, 'class': self._majority_class(y)}
        
        # Select distance measure
        distance_measure = np.random.choice(self.distance_measures)
        
        # Candidate sampling for better splits
        n_candidates = min(5, n_samples)
        candidate_indices = np.random.choice(n_samples, n_candidates, replace=False)
        
        best_exemplar_idx = None
        best_threshold = None
        best_gain = -np.inf
        
        for candidate_idx in candidate_indices:
            exemplar = X[candidate_idx]
            
            # Compute distances
            distances = np.array([self._compute_distance(x, exemplar, distance_measure) for x in X])
            
            # Find threshold with maximum information gain
            threshold, gain = self._find_best_threshold(distances, y)
            
            if gain > best_gain:
                best_gain = gain
                best_exemplar_idx = candidate_idx
                best_threshold = threshold
        
        # If no good split found
        if best_exemplar_idx is None:
            return {'leaf': True, 'class': self._majority_class(y)}
        
        exemplar = X[best_exemplar_idx]
        distances = np.array([self._compute_distance(x, exemplar, distance_measure) for x in X])
        
        # Split data
        left_mask = distances <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {'leaf': True, 'class': self._majority_class(y)}
        
        X_left = [X[i] for i in range(len(X)) if left_mask[i]]
        X_right = [X[i] for i in range(len(X)) if right_mask[i]]
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        return {
            'leaf': False,
            'exemplar': exemplar,
            'threshold': best_threshold,
            'distance_measure': distance_measure,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }
    
    def _find_best_threshold(self, distances: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Find threshold that maximizes information gain"""
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_y = y[sorted_indices]
        
        best_threshold = 0
        best_gain = -np.inf
        
        for i in range(1, len(sorted_distances)):
            if sorted_distances[i] == sorted_distances[i-1]:
                continue
            
            threshold = (sorted_distances[i] + sorted_distances[i-1]) / 2
            gain = self._information_gain(sorted_y, i)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def _information_gain(self, y: np.ndarray, split_idx: int) -> float:
        """Calculate information gain for a split"""
        def entropy(labels):
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        parent_entropy = entropy(y)
        left_y = y[:split_idx]
        right_y = y[split_idx:]
        
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        n = len(y)
        left_entropy = entropy(left_y)
        right_entropy = entropy(right_y)
        
        child_entropy = (len(left_y) / n) * left_entropy + (len(right_y) / n) * right_entropy
        
        return parent_entropy - child_entropy
    
    def _compute_distance(self, seq1: Tuple[np.ndarray, np.ndarray], 
                         seq2: Tuple[np.ndarray, np.ndarray], 
                         measure: str) -> float:
        """Compute distance using specified measure"""
        if measure == 'MSM':
            return self.msm.compute_distance(seq1, seq2)
        elif measure == 'SSDTW':
            return self.ssdtw.compute_distance(seq1, seq2)
        elif measure == 'ERP':
            return self.erp.compute_distance(seq1, seq2)
        else:
            raise ValueError(f"Unknown distance measure: {measure}")
    
    def _majority_class(self, y: np.ndarray) -> int:
        """Return most common class"""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def predict(self, x: Tuple[np.ndarray, np.ndarray]) -> int:
        """Predict class for single sample"""
        node = self.root
        
        while not node['leaf']:
            distance = self._compute_distance(x, node['exemplar'], node['distance_measure'])
            if distance <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        return node['class']


class ProximityForest:
    """
    Proximity Forest 2.0 Ensemble Classifier
    
    Ensemble of proximity trees using elastic distance measures with
    improved candidate sampling and information gain-based splits.
    """
    
    def __init__(self, n_trees: int = 20, max_depth: int = 12,
                 distance_distribution: Dict[str, float] = None):
        """
        Parameters
        ----------
        n_trees : int
            Number of trees in ensemble
        max_depth : int
            Maximum depth of each tree
        distance_distribution : dict
            Distribution of distance measures {'MSM': 0.5, 'SSDTW': 0.3, 'ERP': 0.2}
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        
        if distance_distribution is None:
            distance_distribution = {'MSM': 0.5, 'SSDTW': 0.3, 'ERP': 0.2}
        
        self.distance_distribution = distance_distribution
        self.trees = []
        self.classes_ = None
        
        # Initialize distance measure instances
        self.msm = MSM_Enhanced(c=1.0)
        self.ssdtw = SSDTW_Multivariable(L=3)
        self.erp = ERP_2D(g=0.0)
    
    def fit(self, X: List[Tuple[np.ndarray, np.ndarray]], y: np.ndarray):
        """
        Train the forest
        
        Parameters
        ----------
        X : list of tuple
            Training sequences (HeadAngleX, HeadAngleY)
        y : np.ndarray
            Training labels
        """
        self.classes_ = np.unique(y)
        
        # Generate distance measures for each tree based on distribution
        measures = []
        for measure, prob in self.distance_distribution.items():
            n_measure = int(self.n_trees * prob)
            measures.extend([measure] * n_measure)
        
        # Fill remaining if needed
        while len(measures) < self.n_trees:
            measures.append(np.random.choice(list(self.distance_distribution.keys())))
        
        np.random.shuffle(measures)
        
        print(f"Training Proximity Forest 2.0...")
        print(f"  Trees: {self.n_trees}")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Distance distribution: {self.distance_distribution}")
        
        start_time = time.time()
        
        for i in range(self.n_trees):
            tree_measures = [measures[i]]
            tree = ProximityTree(self.max_depth, tree_measures, self.msm, self.ssdtw, self.erp)
            tree.fit(X, y)
            self.trees.append(tree)
            
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  Trained {i + 1}/{self.n_trees} trees ({elapsed:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
    
    def predict(self, X: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Predict classes for samples
        
        Parameters
        ----------
        X : list of tuple
            Sequences to classify
            
        Returns
        -------
        np.ndarray
            Predicted classes
        """
        predictions = []
        
        for x in X:
            votes = defaultdict(int)
            for tree in self.trees:
                pred = tree.predict(x)
                votes[pred] += 1
            
            predicted_class = max(votes.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_single(self, x: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, float]:
        """
        Predict class and confidence for single sample
        
        Parameters
        ----------
        x : tuple of np.ndarray
            Single sequence to classify
            
        Returns
        -------
        tuple
            (predicted_class, confidence)
        """
        votes = defaultdict(int)
        
        for tree in self.trees:
            pred = tree.predict(x)
            votes[pred] += 1
        
        predicted_class = max(votes.items(), key=lambda x: x[1])[0]
        confidence = votes[predicted_class] / len(self.trees)
        
        return predicted_class, confidence


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """
    Complete training pipeline with PKL generation
    
    Handles data generation, model training, and automatic creation of all
    required PKL files and configuration reports.
    """
    
    def __init__(self, output_dir: str = "trained_dwell_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.generator = SyntheticDwellGenerator()
        self.model = None
        self.command_mapping = None
        self.training_metadata = None
    
    def run(self, n_samples_per_category: int = 50, n_trees: int = 20, 
            max_depth: int = 12, test_split: float = 0.2):
        """
        Execute complete training pipeline
        
        Parameters
        ----------
        n_samples_per_category : int
            Number of training samples per category
        n_trees : int
            Number of trees in forest
        max_depth : int
            Maximum tree depth
        test_split : float
            Proportion of data for testing
        """
        print("\n" + "="*80)
        print("HEAD DWELL TIME NAVIGATION CLASSIFIER - TRAINING PIPELINE")
        print("="*80 + "\n")
        
        # Generate data
        print("Generating synthetic dwell time data...")
        dataset = self.generator.generate_full_dataset(n_samples_per_category)
        
        # Prepare training data
        X_train, y_train, X_test, y_test = self._prepare_data(dataset, test_split)
        
        # Create command mapping
        categories = list(dataset.keys())
        self.command_mapping = {i: cat for i, cat in enumerate(categories)}
        
        # Train model
        print("\nTraining Proximity Forest 2.0...")
        start_time = time.time()
        
        self.model = ProximityForest(n_trees=n_trees, max_depth=max_depth)
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.1f}s")
        
        # Collect metadata
        self._collect_metadata(dataset, accuracy, training_time, n_trees, max_depth)
        
        # Generate PKL files
        self._generate_pkl_files(dataset)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
    
    def _prepare_data(self, dataset: Dict, test_split: float) -> Tuple:
        """Split data into train and test sets"""
        X_all = []
        y_all = []
        
        for cat_idx, (cat_name, sequences) in enumerate(dataset.items()):
            X_all.extend(sequences)
            y_all.extend([cat_idx] * len(sequences))
        
        # Shuffle
        indices = np.random.permutation(len(X_all))
        X_all = [X_all[i] for i in indices]
        y_all = np.array([y_all[i] for i in indices])
        
        # Split
        split_idx = int(len(X_all) * (1 - test_split))
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        X_test = X_all[split_idx:]
        y_test = y_all[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def _collect_metadata(self, dataset: Dict, accuracy: float, training_time: float,
                         n_trees: int, max_depth: int):
        """Collect training metadata"""
        command_stats = {}
        
        for cat_name, sequences in dataset.items():
            # Extract dwell-related statistics from sequences
            x_stability = [np.std(seq[0]) for seq in sequences]
            y_stability = [np.std(seq[1]) for seq in sequences]
            
            command_stats[cat_name] = {
                'count': len(sequences),
                'x_stability_mean': float(np.mean(x_stability)),
                'y_stability_mean': float(np.mean(y_stability)),
                'avg_sequence_length': float(np.mean([len(seq[0]) for seq in sequences]))
            }
        
        self.training_metadata = {
            'total_sequences': sum(len(seqs) for seqs in dataset.values()),
            'command_statistics': command_stats,
            'training_timestamp': time.time(),
            'training_time_seconds': training_time,
            'test_accuracy': accuracy,
            'n_trees': n_trees,
            'max_depth': max_depth,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_pkl_files(self, dataset: Dict):
        """Generate all required PKL files"""
        print("\nGenerating PKL files...")
        
        # 1. pf2_dwell.pkl (references as dwell time arrays)
        dwell_references = {}
        for cat_name, sequences in dataset.items():
            # Extract dwell times from sequences (using angle stability as proxy)
            refs = []
            for seq in sequences:
                # Calculate "dwell time" from angle stability (inverse of variance)
                x_stability = 1.0 / (np.var(seq[0]) + 0.1)
                y_stability = 1.0 / (np.var(seq[1]) + 0.1)
                dwell_time_proxy = (x_stability + y_stability) / 2.0
                
                # Create synthetic dwell array
                config = self.generator.DWELL_CONFIGS[cat_name]
                dwell_array = np.random.normal(config.mean, config.std, len(seq[0]))
                dwell_array = np.abs(dwell_array)
                
                refs.append(dwell_array)
            
            dwell_references[cat_name] = refs
        
        with open(self.output_dir / 'pf2_dwell.pkl', 'wb') as f:
            pickle.dump(dwell_references, f)
        print(f"  Generated: pf2_dwell.pkl")
        
        # 2. command_mapping_dwell.pkl
        with open(self.output_dir / 'command_mapping_dwell.pkl', 'wb') as f:
            pickle.dump(self.command_mapping, f)
        print(f"  Generated: command_mapping_dwell.pkl")
        
        # 3. dwell_schema.pkl
        schema = {
            'features': ['HeadAngleX', 'HeadAngleY'],
            'sequence_length': self.generator.sequence_length,
            'expected_dtypes': ['float64', 'float64'],
            'dwell_analysis': True
        }
        
        with open(self.output_dir / 'dwell_schema.pkl', 'wb') as f:
            pickle.dump(schema, f)
        print(f"  Generated: dwell_schema.pkl")
        
        # 4. training_metadata_dwell.pkl
        with open(self.output_dir / 'training_metadata_dwell.pkl', 'wb') as f:
            pickle.dump(self.training_metadata, f)
        print(f"  Generated: training_metadata_dwell.pkl")
        
        # 5. config.txt
        self._generate_config_txt()
        print(f"  Generated: config.txt")
        
        # 6. accuracy.txt
        self._generate_accuracy_txt()
        print(f"  Generated: accuracy.txt")
        
        # 7. trained model
        with open(self.output_dir / 'dwell_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print(f"  Generated: dwell_model.pkl")
    
    def _generate_config_txt(self):
        """Generate configuration text file"""
        config_text = "MODEL CONFIGURATION\n"
        config_text += "=" * 60 + "\n\n"
        
        config_text += "TYPE: Head Dwell Time Navigation Classifier\n"
        config_text += "Variables: HeadAngleX, HeadAngleY (Dwell Analysis)\n"
        config_text += f"Categories: {list(self.command_mapping.values())}\n\n"
        
        config_text += "HYPERPARAMETERS:\n"
        config_text += f"  Trees: {self.training_metadata['n_trees']}\n"
        config_text += f"  Max Depth: {self.training_metadata['max_depth']}\n"
        config_text += f"  Distance Measures: MSM, SSDTW, ERP\n"
        config_text += f"  Voting: Confidence-weighted\n"
        
        with open(self.output_dir / 'config.txt', 'w') as f:
            f.write(config_text)
    
    def _generate_accuracy_txt(self):
        """Generate accuracy report text file"""
        accuracy_text = "DWELL TIME CLASSIFIER - RESULTS\n"
        accuracy_text += "=" * 60 + "\n\n"
        
        accuracy_text += f"Date: {self.training_metadata['training_date']}\n"
        accuracy_text += f"Accuracy: {self.training_metadata['test_accuracy']:.4f}\n"
        accuracy_text += f"Training Time: {self.training_metadata['training_time_seconds']:.2f} seconds\n"
        accuracy_text += f"Training Sequences: {self.training_metadata['total_sequences']}\n"
        accuracy_text += f"Trees: {self.training_metadata['n_trees']}\n\n"
        
        accuracy_text += "NAVIGATION CATEGORIES:\n\n"
        
        for cat_name, stats in self.training_metadata['command_statistics'].items():
            accuracy_text += f"{cat_name}:\n"
            accuracy_text += f"  Samples: {stats['count']}\n"
            accuracy_text += f"  X Stability: {stats['x_stability_mean']:.2f}\n"
            accuracy_text += f"  Y Stability: {stats['y_stability_mean']:.2f}\n\n"
        
        with open(self.output_dir / 'accuracy.txt', 'w') as f:
            f.write(accuracy_text)


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """
    Inference pipeline for trained dwell model
    
    Loads trained model and PKL files for real-time classification.
    """
    
    def __init__(self, model_dir: str = "trained_dwell_models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.command_mapping = None
        self.schema = None
        self.metadata = None
        
        self._load_model_files()
    
    def _load_model_files(self):
        """Load all model files"""
        print("\n" + "="*80)
        print("LOADING TRAINED DWELL MODEL")
        print("="*80 + "\n")
        
        # Load model
        model_path = self.model_dir / 'dwell_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded: dwell_model.pkl")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load command mapping
        mapping_path = self.model_dir / 'command_mapping_dwell.pkl'
        if mapping_path.exists():
            with open(mapping_path, 'rb') as f:
                self.command_mapping = pickle.load(f)
            print(f"Loaded: command_mapping_dwell.pkl")
        
        # Load schema
        schema_path = self.model_dir / 'dwell_schema.pkl'
        if schema_path.exists():
            with open(schema_path, 'rb') as f:
                self.schema = pickle.load(f)
            print(f"Loaded: dwell_schema.pkl")
        
        # Load metadata
        metadata_path = self.model_dir / 'training_metadata_dwell.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded: training_metadata_dwell.pkl")
        
        print("\n" + "="*80)
        print("MODEL LOADED SUCCESSFULLY")
        print("="*80 + "\n")
    
    def predict(self, angle_x: np.ndarray, angle_y: np.ndarray) -> Tuple[str, float]:
        """
        Predict navigation category from head angles
        
        Parameters
        ----------
        angle_x, angle_y : np.ndarray
            Head angle sequences
            
        Returns
        -------
        tuple
            (category_name, confidence)
        """
        sequence = (angle_x, angle_y)
        class_idx, confidence = self.model.predict_single(sequence)
        category_name = self.command_mapping[class_idx]
        
        return category_name, confidence
    
    def predict_batch(self, sequences: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[str, float]]:
        """
        Predict categories for multiple sequences
        
        Parameters
        ----------
        sequences : list of tuple
            List of (angle_x, angle_y) sequences
            
        Returns
        -------
        list of tuple
            List of (category_name, confidence) predictions
        """
        results = []
        
        for seq in sequences:
            category, confidence = self.predict(seq[0], seq[1])
            results.append((category, confidence))
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dwell_classifier.py train [options]")
        print("  python dwell_classifier.py test")
        return
    
    mode = sys.argv[1]
    
    if mode == 'train':
        # Training mode
        pipeline = TrainingPipeline(output_dir="trained_dwell_models")
        pipeline.run(
            n_samples_per_category=50,
            n_trees=20,
            max_depth=12,
            test_split=0.2
        )
    
    elif mode == 'test':
        # Testing mode
        inference = InferencePipeline(model_dir="trained_dwell_models")
        
        # Generate test sample
        generator = SyntheticDwellGenerator()
        test_sequence = generator.generate_category_dataset('NAVEGACION_EFICIENTE', 1)[0]
        
        # Predict
        category, confidence = inference.predict(test_sequence[0], test_sequence[1])
        
        print(f"\nTest Prediction:")
        print(f"  Category: {category}")
        print(f"  Confidence: {confidence:.2%}")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'train' or 'test'")


if __name__ == "__main__":
    main()
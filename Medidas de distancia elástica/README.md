A comprehensive collection of 10 elastic distance measures implemented in Python for analyzing VR wheelchair movement patterns.

Overview

This collection implements state-of-the-art elastic distance measures specifically designed for VR research applications. Each algorithm features automatic NumPy operation interception for precise computational analysis, real-time performance monitoring, and comprehensive visualization tools.

Algorithms Included

Dynamic Time Warping (DTW)
- Classic DTW implementation with optimized dynamic programming
- Automatic path recovery and cost matrix visualization
- Real-time FLOP counting for computational analysis
- Applications: Basic temporal alignment of movement trajectories

Weighted Dynamic Time Warping (WDTW)
- Sigmoid weight function implementation: w(|i-j|) = 1/(1 + e^(-g(|i-j| - m/2)))
- Configurable slope parameter for temporal penalty adjustment
- Enhanced cost matrices with weight visualization
- Applications: Penalizing large temporal deviations in movement analysis

Edit Distance with Real Penalty (ERP)
- Gap penalty system for handling missing or irregular data points
- Reference point-based distance calculations
- Robust handling of sequences with different sampling rates
- Applications: Movement data with irregular sampling or missing points

Longest Common Subsequence (LCSS)
- Threshold-based similarity matching algorithm
- Configurable epsilon and delta parameters
- Subsequence extraction and analysis
- Applications: Finding common movement patterns in wheelchair trajectories

Time Warp Edit (TWE)
- Combined edit and warping operations
- Configurable stiffness parameter for time penalty
- Balanced approach between DTW and edit distance
- Applications: Movement analysis requiring both temporal and spatial flexibility

Move-Split-Merge (MSM)
- Advanced three-operation distance measure
- Move, split, and merge cost parameters
- Optimal path computation with operation tracking
- Applications: Complex trajectory transformations and movement pattern analysis

Constrained DTW (cDTW)
- Sakoe-Chiba band constraint implementation
- Window-based alignment restrictions
- Reduced computational complexity for large sequences
- Applications: Fast processing of long movement sequences

Adaptive DTW (ADTW)
- Adaptive warping path constraints
- Dynamic adjustment of alignment flexibility
- Context-aware distance computation
- Applications: Variable-length movement segments with adaptive alignment

Edit Distance on Real Sequences (EDR)
- Threshold-based matching with edit operations
- Real-valued sequence processing
- Configurable matching tolerance
- Applications: Noisy movement data with tolerance-based matching

Incremental Constrained DTW (ICDTW)
- Incremental computation for streaming data
- Memory-efficient processing
- Real-time analysis capabilities
- Applications: Live VR session analysis and real-time feedback

Core Features

Exact FLOP Counting System
- Automatic NumPy operation interception
- Real-time computational cost tracking
- Detailed performance profiling per algorithm
- Operation breakdown analysis (addition, multiplication, exponential, etc.)

Performance Analysis Tools
- Throughput measurement (FLOPs/second)
- Memory usage monitoring
- Execution time profiling
- Comparative efficiency analysis

Visualization Suite
- Cost matrix heatmaps for each algorithm
- Optimal path overlay visualization
- FLOP distribution pie charts
- Performance comparison bar charts
- Algorithm-specific parameter visualization

Data Processing Pipeline
- Automatic CSV data loading and preprocessing
- Multi-participant batch processing
- Hierarchical output organization
- Comprehensive similarity scoring

Requirements

- Python 3.8+
- NumPy (with automatic operation interception)
- Pandas (data processing)
- Matplotlib (visualization)
- Time (performance measurement)

Installation

1. Clone the repository
2. Install required dependencies: pip install numpy pandas matplotlib
3. Import desired algorithms into your Unity

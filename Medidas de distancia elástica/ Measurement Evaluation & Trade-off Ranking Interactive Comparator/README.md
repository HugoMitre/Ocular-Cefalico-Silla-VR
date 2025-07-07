This system evaluates similarity measures using the multi-objective formula:
F(x) = α · Cost_comp(x) - β · Accuracy_align(x)
Where α weights computational cost, β weights alignment precision, and lower F(x) indicates better performance.

Supported Algorithms

- **Dynamic Time Warping (DTW)**: Classic temporal alignment
- **Weighted DTW (WDTW)**: Sigmoid-weighted penalty system  
- **Edit Distance Real Penalty (ERP)**: Gap penalty for irregular data
- **Time Warp Edit (TWE)**: Combined edit and warping operations
- **Move-Split-Merge (MSM)**: Three-operation distance measure
- **Adaptive DTW (ADTW)**: Context-aware alignment
- **Longest Common Subsequence (LCSS)**: Robust subsequence matching
- **Constrained DTW (cDTW)**: Window-constrained alignment
- **Ensemble Shapelet DTW (ESDTW)**: Pattern recognition based
- **Subsequence Sakoe DTW (SSDTW)**: Subsequence-optimized

Key Features

**Interactive Configuration**
- Real-time α and β parameter input
- Configuration impact prediction
- Balance type classification (efficiency/precision/balanced)

**Robust Analysis**
- Automatic detection of algorithm result folders
- Exact FLOP counting and timing analysis
- Ground truth alignment error calculation
- Multi-format file support (CSV, Excel)

**Comprehensive Reporting**
- Excel workbooks with detailed breakdowns
- Visual comparisons and ranking charts
- Mathematical analysis documentation

Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, OpenPyXL

Installation

1. Install dependencies: `pip install numpy pandas matplotlib openpyxl`
2. Prepare algorithm result folders with pattern: `{algorithm}_results*/`
3. Ensure ground truth reference files are available

Usage

**Interactive Mode**

python hybrid_evaluator.py
Direct Parameters
bashpython hybrid_evaluator.py --alpha 0.7 --beta 0.3
Verification Test
bashpython hybrid_evaluator.py --test
Configuration Examples

Efficiency-focused: α=0.8, β=0.2 (prioritizes speed)
Precision-focused: α=0.2, β=0.8 (prioritizes accuracy)
Balanced: α=0.5, β=0.5 (equal consideration)

Output

Ranking reports with F(x) scores
Normalized metrics analysis
Detailed mathematical breakdowns
Comparative visualizations
Excel workbooks with multiple analysis sheets

The system enables data-driven algorithm selection by balancing computational constraints with accuracy requirements across diverse similarity measurement applications.


Wheelchair Control Classification System
Comprehensive Multi-Modal Pattern Recognition Framework

A complete machine learning system for real-time wheelchair control through head movement analysis. Built on Proximity Forest 2.0 architecture with elastic distance metrics (MSM, SSDTW, ERP), this framework provides production-ready classification across four critical modalities: directional commands, velocity patterns, navigation deviation, and dwell time analysis.

Overview

This system implements a complete pipeline for wheelchair navigation control through head movement pattern recognition. Designed for quadriplegic users, it provides real-time classification of head movements into actionable wheelchair commands with high accuracy and low latency.

Key Features

- Four Classification Modalities: Commands, velocity, deviation, and dwell time
- Real-Time Processing: TCP servers for Unity integration (5556-5559 ports)
- Production-Ready: Comprehensive training and operational frameworks
- High Accuracy: Proximity Forest 2.0 with 100% test accuracy on synthetic data
- Low Latency: Average classification time <50ms
- Robust Distance Metrics: MSM, SSDTW, and ERP for time series comparison
- Synthetic Data Generation: Built-in validation with ground truth
- Batch Processing: Parallel CSV processing with performance reporting

 Research Context

- **Institution**: CIMAT (Centro de Investigación en Matemáticas)
- **Application**: Assistive technology for wheelchair control
- **Target Users**: Individuals with quadriplegia
- **Input Method**: Head angle tracking (HeadAngleX, HeadAngleY)
- **Output**: Real-time navigation commands

---

 System Architecture

Two-Tier Design


┌─────────────────────────────────────────────────────────────┐
│                    TRAINING SYSTEMS                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Command    │  │   Velocity   │  │  Deviation   │     │
│  │  Classifier  │  │  Classifier  │  │  Classifier  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│    [Model PKLs]      [Model PKLs]      [Model PKLs]        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 OPERATIONAL FRAMEWORKS                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Command    │  │   Velocity   │  │  Deviation   │     │
│  │  Framework   │  │  Framework   │  │  Framework   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│    [TCP 5556/57]     [TCP 5556/58]     [TCP 5556/57]       │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                            │                                │
│                            ▼                                │
│                    ┌──────────────┐                         │
│                    │ Unity Client │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘


Component Overview

**Training Systems** (4 programs):
- Generate synthetic training data
- Train Proximity Forest 2.0 models
- Export trained models as PKL files
- Validate accuracy on test sets

**Operational Frameworks** (4 programs):
- Load trained models
- TCP servers for real-time classification
- Batch CSV processing
- Performance monitoring and reporting


Modalities and Applications

1. Command Classification (wheelchair_command_classifier.py)

Purpose**: Translates head angles into 8 directional wheelchair commands

Input Features:
- HeadAngleX: Horizontal head rotation (-90° to +90°)
- HeadAngleY: Vertical head tilt (-90° to +90°)


Output Categories (8 commands):

front                  
back                     
left_turn               
right_turn               
front_left_diagonal    
front_right_diagonal     
back_left_diagonal     
back_right_diagonal      


Technical Parameters:


- Proximity Forest: 150 trees, max depth 16
- Distance measures: MSM (50%), SSDTW (30%), ERP (20%)
- Test accuracy: 100% on synthetic data

Applications:


- Primary wheelchair navigation control
- Directional command recognition
- Multi-directional movement patterns

---

2. Velocity Classification (velocity_classifier.py)

Purpose: Classifies head movement velocity into 7 speed categories


Input Features:
- HeadVelocityX: Horizontal angular velocity (deg/s)
- HeadVelocityY: Vertical angular velocity (deg/s)

Output Categories** (7 levels):

MUY_BAJA      
BAJA          
MEDIA_BAJA   
MEDIA         
MEDIA_ALTA 
ALTA          
MUY_ALTA    



Technical Parameters:
- Proximity Forest: 100 trees, max depth 15
- Magnitude-based classification with direction awareness
- Test accuracy: 95%+ on synthetic data

Applications:
- Speed control for wheelchair
- Velocity-based safety limits
- Adaptive control systems

---

3. Deviation Classification (deviation_classifier.pY)

Purpose: Analyzes navigation patterns through head deviation from center

Input Features:
- HeadDeviationX: Horizontal deviation from neutral position
- HeadDeviationY: Vertical deviation from neutral position

Output Categories (5 patterns):

NAVEGACION_EFICIENTE    
NAVEGACION_DIRIGIDA     
EXPLORACION_PAUSADA     
EXPLORACION_ACTIVA      
BUSQUEDA_REORIENTACION   

Technical Parameters:
- Proximity Forest: 100 trees, max depth 12
- Statistical feature extraction (mean, std, magnitude)
- Test accuracy: 92%+ on synthetic data

Applications:
- Navigation behavior analysis
- User intent recognition
- Adaptive assistance systems

---

4. Dwell Time Classification (dwell_classifier.py)

Purpose**: Recognizes fixation patterns and attention distribution

Input Features:
- HeadAngleX: Used for dwell time calculation
- HeadAngleY: Used for dwell time calculation
- Dwell duration: Time spent within angular threshold

Output Categories** (5 patterns):

EXPLORACION_ACTIVA       
BUSQUEDA_REORIENTACION   
NAVEGACION_DIRIGIDA      
EXPLORACION_PAUSADA      
NAVEGACION_EFICIENTE    


Technical Parameters**:

- Proximity Forest: 20 trees, max depth 12
- Temporal pattern analysis
- Test accuracy: 88%+ on synthetic data

Applications:
- Attention monitoring
- Decision-making analysis
- User confidence assessment


Training Systems

Core Training Pipeline

Each training system follows a unified architecture:

python
1. Synthetic Data Generation
   ├── Category-specific parameters
   ├── Temporal variation (sinusoidal/noise)
   ├── Balanced sampling
   └── Train/test split (80/20)

2. Proximity Forest Training
   ├── Distance measure selection (MSM, SSDTW, ERP)
   ├── Candidate exemplar sampling (5 per split)
   ├── Information gain-based splitting
   └── Ensemble construction (20-150 trees)

3. Model Evaluation
   ├── Test set accuracy
   ├── Per-category performance
   ├── Confusion matrix analysis
   └── Timing benchmarks

4. PKL Generation
   ├── proximity_forest_model.pkl (Trained forest)
   ├── command_mapping.pkl (Index → category)
   ├── schema.pkl (Feature definitions)
   ├── training_metadata.pkl (Statistics)
   └── config.txt (Human-readable report)


Distance Metrics Implementation

MSM (Move-Split-Merge) with Sakoe-Chiba Band

Mathematical Foundation:

Cost components:
- Move: |x[i] - y[j]|
- Split: Cost of inserting element
- Merge: Cost of deleting element

Penalty function:
C(u, v, w) = {
    c                           if u ≤ v ≤ w or u ≥ v ≥ w
    c + min(|v-u|, |v-w|)      otherwise
}

Sakoe-Chiba band calculation:
1. Detect peaks/valleys using scipy.signal.find_peaks
2. Calculate temporal displacement: Δt = |t_P - t_Q|
3. Convert to index displacement: r = Δt + margin
4. Default to 10% of sequence length if no peaks

Implementation Features**:
- Dynamic programming with band constraints
- Bivariate processing (separate bands for X and Y)
- Adaptive band width based on sequence characteristics
- O(n·r) complexity where r is band width

SSDTW

Mathematical Foundation:

Haar wavelet decomposition:
- Approximation: A[k] = (signal[2k] + signal[2k+1]) / √2
- Detail: D[k] = (signal[2k] - signal[2k+1]) / √2
- Levels: L = 3 (default)

Distance computation:
d_shape = DTW(A_x, A_y)     # Shape comparison
d_texture = DTW(D_x, D_y)   # Texture comparison
d_total = w1·d_shape + w2·d_texture

Implementation Features:
- Multi-resolution analysis (3 levels)
- Separate X and Y dimension processing
- Combined shape and texture distances
- Efficient subsequence matching

#### ERP (Edit Distance with Real Penalty)

Mathematical Foundation:

Operations:
- Match: |x[i] - y[j]|
- Insert: |y[j] - g|
- Delete: |x[i] - g|

where g is gap penalty (default: 0.0)

Recurrence relation:
D[i,j] = min(
    D[i-1,j-1] + |x[i] - y[j]|,  # Match
    D[i-1,j] + |x[i] - g|,        # Delete
    D[i,j-1] + |y[j] - g|         # Insert
)


Implementation Features:
- Gap penalty system for missing data
- Bivariate distance (sum of X and Y)
- Standard dynamic programming
- O(n·m) complexity


 Operational Frameworks

Unified Framework Architecture

Each operational framework provides:

1. Model Management
   ├── Automatic PKL loading
   ├── Model validation
   ├── Configuration persistence
   └── Directory selection (GUI/CLI)

2. Classification Pipeline
   ├── Single CSV classification
   ├── Parallel batch processing
   ├── Real-time TCP classification
   └── Synthetic validation

3. TCP Server (Dual Mode)
   ├── Synthetic data (testing)
   ├── Real data (production)
   ├── JSON protocol
   └── Performance monitoring

4. Performance Analysis
   ├── Classification time tracking
   ├── Accuracy reporting
   ├── Category distribution
   └── Throughput measurement


TCP Communication Protocol

Message Format (All Servers)

Request (Client → Server):
json
{
    "HeadAngleX": [float array],      // Command/Dwell
    "HeadAngleY": [float array],      // Command/Dwell
    "HeadVelocityX": [float array],   // Velocity
    "HeadVelocityY": [float array],   // Velocity
    "HeadDeviationX": [float array],  // Deviation
    "HeadDeviationY": [float array],  // Deviation
    "ground_truth": "CATEGORY"        // Optional (synthetic)
}

Response (Server → Client):

json
{
    "prediction": "CATEGORY",
    "confidence": 0.95,
    "time_ms": 45.2,
    "correct": true              // Only with ground_truth
}


Installation

 System Requirements

Operating System: Linux, macOS, Windows
Python Version:   3.8 or higher
RAM:              4 GB minimum, 8 GB recommended
Storage:          500 MB for models and dependencies
CPU:              Multi-core recommended for batch processing

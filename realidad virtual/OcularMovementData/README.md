A gaze tracking and analysis system for Unity that monitors eye movement direction and velocity in VR environments. The system uses LineRenderer components and deviation calculations to analyze gaze behavior patterns with real-time data processing and CSV export capabilities.
Requirements

Unity 2021.3 LTS or higher
C# scripting support
LineRenderer component
VR-compatible device (optional)

Installation

Import all .cs scripts into your Unity project
Attach the scripts to the LineRenderer component connected to the eye tracking system
Configure LineRenderer settings in the gaze tracking scripts
Ensure proper component linking in the Unity Inspector

Usage
System Operation

Run the main scene in Unity
Verify LineRenderer is properly configured
The system automatically begins real-time gaze tracking
Data is analyzed and stored in CSV files
Use Debug.Log() to monitor console output if needed

Data Collection

Real-time Processing: Continuous gaze direction and velocity monitoring
Automatic Storage: Data exported to C:\Users\Manuel Delgado\Documents\
File Management: Duplicate files receive numeric suffixes or timestamps

Core Scripts
GazeDataLogger.cs
Primary data collection component:

Records gaze direction and angular velocity
Applies moving average filter for data smoothing
Normalizes velocity values within predefined ranges
Exports data to CSV upon simulation completion

GazeDirectionWithArea.cs
Spatial gaze analysis system:

Detects gaze location within screen zones
Classifies gaze direction into predefined categories
Uses BoxCollider components to define areas of interest
Records directional data in CSV format

GazeStandardDeviation.cs
Statistical variance analyzer:

Calculates standard deviation of gaze direction
Uses data windows for variation analysis
Normalizes deviation values for comparison
Stores statistical values in CSV files

NormalizedGazeDwellTimeCalculator.cs
Advanced dwell time calculator:

Measures gaze persistence in different areas
Implements adaptive Kalman filtering for enhanced accuracy
Uses circular buffer for real-time data analysis
Exports dwell time data for statistical analysis

Data Output
CSV Structure
Exported files contain:

Temporal: Timestamp data
Velocity: Gaze velocity (X,Y coordinates)
Statistical: Standard deviation (X,Y coordinates)
Directional: Gaze direction classification
Behavioral: Dwell time measurements

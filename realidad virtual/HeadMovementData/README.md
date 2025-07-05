A comprehensive head tracking and analysis system for Unity that monitors head direction and angular velocity in real-time. The system calculates horizontal and vertical angles relative to a reference point to classify head direction into different categories. It analyzes angular velocity with data normalization and stores all information in CSV files for research analysis.
Requirements

Unity 2021.3 LTS or higher
C# scripting support
Camera.main component
VR-compatible device (optional)

Installation

Import all .cs scripts into your Unity project
User-friendly setup: The system works based on camera rotation, requiring only the main camera
Configure the direction reference object (frontReference) in the Unity Inspector
No additional components needed beyond the standard camera

Usage
System Operation

Run the main scene in Unity
Set up the reference object (frontReference) in the scene
The system automatically calculates real-time head direction and angular velocity
Data is recorded in lists and saved to CSV files upon simulation completion
Use Debug.Log() to monitor console output if needed

Data Processing

Real-time Calculations: Continuous head direction and velocity monitoring
Automatic Storage: Data exported to C:\Users\Manuel Delgado\Documents\
File Management: Duplicate files receive numeric suffixes or timestamps

Core Scripts
HeadDirectionTracker.cs
Primary direction analysis component:

Calculates horizontal and vertical angles relative to frontal reference
Determines head direction based on predefined thresholds
Stores head direction data in lists and CSV files
Supports customizable sampling intervals and angle thresholds

StandardDeviationCalculator.cs
Statistical variance analyzer:

Calculates real-time standard deviation of head direction
Applies moving average filter for improved data accuracy
Normalizes deviation values for comparative analysis
Stores statistical values in CSV files for further analysis

NormalizedDwellTimeCalculator.cs
Dwell time measurement system:

Calculates head persistence time in different areas
Defines stability threshold to detect prolonged movements in same position
Normalizes dwell time for easier interpretation
Saves dwell time data in CSV format for statistical analysis

AngularVelocityCalculator.cs
Velocity analysis component:

Calculates head angular velocity on horizontal and vertical axes
Applies moving average filter to smooth abrupt variations
Normalizes angular velocity values for comparative analysis
Stores values in CSV with detailed time and velocity data

Data Output
CSV Structure
Exported files contain:

Temporal: Timestamp information
Angular: Horizontal and vertical angles
Directional: Head direction classification
Velocity: Angular velocity (X/Y axes)
Statistical: Standard deviation measurements
Behavioral: Dwell time data

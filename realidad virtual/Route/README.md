Unity Path Generation and Tracking System

A comprehensive path generation and tracking system for Unity that uses LineRenderer and Bezier algorithms to evaluate wheelchair movement deviation in virtual environments. The system captures precise data for detailed analysis of movement patterns and route adherence.

Requirements

- Unity 2021.3 LTS or higher
- C# scripting support
- LineRenderer component library
- VR-compatible device (optional)

Installation

1. Import all .cs files from this repository into your Unity project
2. Ensure proper component dependencies are met
3. Configure scene objects according to project requirements

Usage

System Operation
1. Scene Execution: Run the main scene in Unity
2. Wheelchair Setup: Place wheelchair in scene and define starting point
3. Route Generation: Observe automatic route generation and deviation calculations
4. Real-time Analysis: Monitor deviation data within Unity Editor
5. Data Export: Save generated data to CSV files for subsequent analysis

Data Collection
- Automatic Recording: Position and deviation data captured at regular intervals
- File Storage: Data exported to C:\Users\Manuel Delgado\Documents\ (customize path as needed)
- File Management: Duplicate files receive numeric suffixes or timestamps

Core Scripts

CustomCurve.cs
Bezier curve generation system:
- Anchor Points: Defines control points for generating smooth Bezier curves
- Real-time Visualization: Uses LineRenderer to display trajectory paths
- Configurable Parameters: Adjustable smoothing and line width settings
- Dynamic Rendering: Updates curve visualization in real-time

RutaManager.cs
Primary route management component:
- Position Tracking: Records real and ideal wheelchair positions
- Deviation Calculation: Computes deviation from predefined scene routes
- Automatic Export: Saves data to CSV files upon simulation completion
- Performance Monitoring: Tracks movement accuracy and path adherence

zonas_visual.cs
Zone detection and interaction system:
- Area Detection: Identifies when wheelchair enters specific scene zones
- Task Registration: Records tasks and commands based on activated zones
- Data Integration: Interacts with DataCombiner to update activity records
- Event Logging: Maintains detailed zone interaction history

Autopathfolower.cs
Automated path following system:
- Route Detection: Automatically identifies paths to follow
- Internal Processing: Executes route traversal through internal algorithms
- Line Following: Maintains adherence to designated path lines
- Autonomous Navigation: Provides automated wheelchair movement along routes

EtapaAudioManager.cs
Multi-stage audio management system:
- Stage-based Audio: Manages different audio tracks for various experiment stages
- Audio Assignment: Allows selection of specific audio for each stage
- Participant Integration: Works with participant selector to determine current stage
- Synchronized Playback: Ensures appropriate audio plays during corresponding experiment phases

Data Output

CSV Structure
Exported files contain:
- Temporal: Timestamp information
- Real Position: Actual wheelchair coordinates (X,Y,Z)
- Ideal Position: Target path coordinates (X,Z)
- Deviation Metrics: Calculated deviation values
- Zone Activity: Area interaction records
- Audio Events: Stage and audio transition logs

File Organization
Documents/
├── wheelchair_tracking_[timestamp].csv
├── route_deviation_[timestamp].csv
├── zone_interactions_[timestamp].csv
└── audio_stages_[timestamp].csv

Configuration

Data Collection Settings
- Sampling Interval: Adjust in RutaManager.cs for data recording frequency
- Route Visualization: Modify smoothing values in CustomCurve.cs for path display
- Zone Sensitivity: Configure detection parameters in zonas_visual.cs

Audio Management
- Stage Assignment: Configure audio tracks for different experiment phases
- Participant Sync: Ensure audio manager reads correct stage from participant selector
- Playback Control: Set audio timing and transition parameters

Debug and Monitoring
- Enable Debug.Log() in RutaManager.cs and zonas_visual.cs for console monitoring
- Use Unity's Scene view to visualize path generation in real-time
- Monitor performance metrics during extended tracking sessions

Best Practices

Setup and Calibration
1. Route Definition: Carefully define ideal paths before data collection
2. Zone Configuration: Set up detection zones with appropriate boundaries
3. Audio Preparation: Test all audio tracks before experiment sessions
4. Wheelchair Positioning: Ensure accurate starting position setup

Data Quality Assurance
1. Validation Testing: Run test sessions to verify tracking accuracy
2. Performance Monitoring: Check system performance during operation
3. Data Backup: Regularly backup CSV files and organize by session
4. Error Handling: Monitor console for any tracking anomalies

Experiment Management
1. Stage Coordination: Synchronize audio stages with experiment phases
2. Participant Workflow: Ensure smooth transition between different stages
3. Data Organization: Maintain clear file naming conventions
4. Session Documentation: Keep detailed records of experiment parameters

Troubleshooting

Common Issues
- Path Visualization: Check LineRenderer component configuration
- Zone Detection: Verify collider setup for detection areas
- Audio Sync: Confirm participant selector stage reading
- Data Export: Ensure write permissions for output directory

Performance Optimization
- Monitor frame rates during path generation
- Optimize Bezier curve calculation frequency
- Check memory usage with multiple simultaneous tracking systems
- Validate file I/O performance during data export

This system provides comprehensive wheelchair movement analysis capabilities for research applications, combining precise path tracking with multi-modal data collection and automated route following functionality.

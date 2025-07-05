Unity Multi-Data Tracking System
Description
A comprehensive real-time data collection system for Unity VR applications. It captures head tracking, gaze analysis, angular velocity, spatial deviation, dwell time, and wheelchair positioning data. All information is automatically organized into CSV files with participant-specific folders for efficient analysis.
Requirements

Unity 2021.3 LTS or higher
C# scripting support
Camera.main component
CSV file system access
Unity Editor with PlayerPrefs

Installation

Import all .cs scripts into your Unity project
Important: Place ParticipantSelector script in an Editor folder (exact name required)
Configure tracking objects in the Inspector to link cameras, wheelchair, and tracking scripts

Usage
Starting the System

Run the main scene in Unity
The system automatically begins real-time data collection
Data saves to: C:\Users\Manuel Delgado\Documents\VR_Study\ (customize as needed)

Controls

S key: Manual data save
D key: View data summary in Unity Console
Spacebar: Toggle cybersickness state (0=none, 1=present)

Core Scripts
DataCombiner.cs
Central data orchestrator that:

Combines multiple tracking data streams
Organizes data into structured CSV format
Manages task and command updates
Creates participant folder structures

ParticipantSelector.cs
Unity Editor window for:

Selecting participant number and attempt
Saving configuration with PlayerPrefs
Auto-generating folder hierarchy

CybersicknessRecorder.cs
Motion sickness tracking system:

Records binary sickness state via spacebar toggle
Captures data at configurable intervals (default: 0.2s)
Provides visual feedback on screen
Integrates with DataCombiner for correlation analysis

Data Output
CSV Structure
Exported files include:

Temporal: Timestamps and elapsed time
Spatial: Position coordinates and angles
Motion: Angular velocity and acceleration
Behavioral: Gaze direction and dwell time
Control: Wheelchair input and user actions
Health: Cybersickness episodes

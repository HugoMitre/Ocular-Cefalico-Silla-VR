A comprehensive control system for Unity that supports both keyboard input and physical joystick integration through Arduino and WebSocket communication. The system enables real-time data transmission for VR environments, simulations, and interactive control systems.
Keyboard Controls
The system responds to the following keyboard inputs:

W: Forward movement
S: Backward movement
A: Turn left
D: Turn right

Physical Joystick Implementation
Hardware Requirements

STM32 microcontroller
Analog joystick
USB cable for connection

Arduino Setup Process

Open Arduino IDE
Load Code: Import the "Joystick_Arduino" code
Configure Tools Menu:

Select the correct STM32 model
Configure parameters according to your microcontroller


Upload: Compile and upload code to the microcontroller

Arduino-Unity Communication Bridge
System Description
This project implements a real-time communication bridge between Arduino and Unity using WebSockets. It receives joystick data from Arduino via serial port and transmits it to Unity for integration in VR environments, simulations, or interactive control systems.
Core Functionality
Arduino Detection and Connection

Automatic Port Detection: Searches for Arduino connection automatically
Serial Communication: Establishes connection at 9600 baud rate
Auto-Reconnection: Attempts automatic reconnection if device disconnects

WebSocket Server Initialization

Server Setup: Starts server at ws://localhost:8080
Multi-Client Support: Allows multiple simultaneous client connections
Real-Time Management: Handles client connections and disconnections in real-time

Data Reception and Transmission

Data Format: Reads Arduino data as Angle,Magnitude
Data Validation: Ensures correct ranges (0°-360° for angle, 0%-100% for magnitude)
Broadcasting: Transmits valid data to all connected WebSocket clients (including Unity)
Console Monitoring: Displays connection status and transmitted values

Event Logging and Error Handling

Log Generation: Creates log files for connection events, disconnections, and errors
Error Management: Handles serial and WebSocket communication errors for system stability

System Requirements
Hardware

Arduino with analog joystick or sensor sending data in Angle,Magnitude format

Software

Python 3.8+
Required Libraries:
bashpip install asyncio websockets pyserial

Unity: For receiving WebSocket data

Implementation Instructions
Step 1: Arduino Setup

Connect Arduino to PC
Ensure Arduino sends data in Angle,Magnitude format via serial port

Step 2: Python Bridge Execution

Run the bridge script on the PC with connected Arduino:
bashpython bridge.py


Step 3: Unity WebSocket Integration
Install WebSocket Plugin

Plugin Requirement: Install WebSocket plugin compatible with your Unity version
File Location: Find appropriate .dll file in websocket/lib folder
Version Selection: For .NET Standard, navigate to standard folder and copy the .dll file
Unity Integration: Place .dll file in Unity's Plugins folder for system recognition

Create Unity Script

Script Creation: Create new C# script named MyMessage.cs in Unity
Code Implementation: Paste the corresponding code into the script
Compilation Check: Verify no compilation errors appear (indicates correct plugin installation)

Assign Script to GameObject

Validation: Confirm script compiles without errors
Assignment: Attach script to GameObject representing the user or entity to be moved in the scene
Connection: Unity establishes connection with WebSocket server at ws://localhost:8080 for real-time joystick data

Best Practices
Hardware Setup

Stable Connections: Ensure secure Arduino and joystick connections
Power Management: Verify adequate power supply for all components
Testing: Validate joystick functionality before integration

Software Configuration

Port Verification: Confirm correct Arduino port detection
Data Validation: Monitor console output for proper data transmission
Error Handling: Check log files for connection issues
Performance: Monitor system performance during operation

Unity Integration

Plugin Compatibility: Verify WebSocket plugin matches Unity version
Script Validation: Test script compilation before scene integration
GameObject Assignment: Ensure proper script attachment to target objects
Connection Testing: Validate WebSocket connection establishment

Troubleshooting
Common Issues

Arduino Connection: Check USB cable and port selection
WebSocket Errors: Verify Python libraries installation
Unity Compilation: Confirm correct plugin placement in Plugins folder
Data Format: Ensure Arduino sends data in correct Angle,Magnitude format

Performance Optimization

Monitor serial communication stability
Check WebSocket server performance with multiple clients
Validate real-time data transmission rates
Test system responsiveness under various conditions

This system provides robust real-time control capabilities for Unity applications with both traditional keyboard input and advanced physical joystick integration.

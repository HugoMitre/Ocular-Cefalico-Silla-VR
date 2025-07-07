EtapaAudioManager - Dynamic Audio Management System for Unity

A dynamic audio management system for Unity that automatically plays specific audio clips based on the current game stage. This component enables a personalized audio experience that adapts to player progress.

Key Features

- Stage-based audio playback: Plays different audio clips according to current game stage
- Flexible configuration: Allows assignment of specific audio clips for each stage
- Configurable initial delay: Controls wait time before audio playback
- Automatic AudioSource management: Automatically creates AudioSource if none exists
- Logging system: Provides detailed information about playback and potential errors

System Requirements

Software
- Unity 2019.4 LTS or higher
- Scripts with .NET Standard 2.1 or higher

Dependencies
- UnityEngine (included in Unity)
- System.Collections (included in Unity)

Component Setup

Step 1: Unity Integration

1. Create the script:
  - Create a new C# script called EtapaAudioManager.cs
  - Copy the component code into the file

2. Verify compilation:
  - Ensure no compilation errors appear
  - Confirm the script appears in the Inspector

Step 2: GameObject Assignment

1. Select GameObject:
  - Choose the GameObject that will handle stage audio
  - Recommended: A dedicated empty GameObject or the main GameManager

2. Attach script:
  - Drag the EtapaAudioManager script to the selected GameObject
  - Or use "Add Component" → search "EtapaAudioManager"

Step 3: Audio Clip Configuration

Required Audio Clips
- Stage 1 Audio: Complete instructions for the first stage
- Stage 2 Audio: Brief instructions for the second stage
- Stage 3 Audio: Brief instructions for the third stage

Inspector Assignment
1. Expand the "Audio Clips" section
2. Drag corresponding audio clips to each slot:
  - audioEtapa1 → Audio for Stage 1
  - audioEtapa2 → Audio for Stage 2
  - audioEtapa3 → Audio for Stage 3

Step 4: Parameter Configuration

Basic Configuration
- Initial Delay: Time in seconds before playing audio (default: 2.0f)
- Audio Source: AudioSource component (automatically created if none exists)

System Functionality

Automatic Stage Detection
The system uses PlayerPrefs to detect the current stage:
- Key: "EtapaActual"
- Default value: 1
- Supported range: 1-3

Playback Flow
1. Initialization: Verifies and configures AudioSource
2. Delay: Waits for the time configured in initialDelay
3. Detection: Gets current stage from PlayerPrefs
4. Selection: Chooses corresponding audio clip
5. Playback: Plays selected audio

Error Management
- Missing audio: Shows console error if no clip is assigned
- Unrecognized stage: Uses Stage 1 audio as fallback
- Missing AudioSource: Automatically creates AudioSource component

Implementation Best Practices

Asset Organization
- Create an Audio/Stages/ folder to organize clips
- Name files descriptively: Stage1_Instructions.wav
- Use optimized formats (WAV for high quality, MP3 for smaller size)

Audio Configuration
- Recommended format: WAV for quality, MP3 for optimization
- Import configuration: Configure according to project needs
- Volume: Adjust in AudioSource for consistency

Stage Management
- Update PlayerPrefs.SetInt("EtapaActual", stageNumber) when stage changes
- Consider using an event system to synchronize stage changes

Common Troubleshooting

Audio not playing
- Verify assignment: Confirm clips are assigned in Inspector
- Verify AudioSource: Ensure AudioSource is not muted
- Verify PlayerPrefs: Confirm "EtapaActual" has a valid value

Incorrect stage
- Verify PlayerPrefs: Use PlayerPrefs.GetInt("EtapaActual") in Inspector
- Verify synchronization: Ensure stage change is saved correctly

Compilation failed
- Verify using statements: Confirm all dependencies are included
- Verify Unity version: Ensure using Unity 2019.4 LTS or higher

Performance Optimization

Memory Management
- Audio clips are loaded on demand
- No unnecessary references maintained in memory
- AudioSource is reused for all clips

Runtime Performance
- Minimal performance impact during playback
- Optimized PlayerPrefs operations for single call
- Conditional logging for production builds

Extensibility

Adding More Stages
1. Add new AudioClip variables to script
2. Expand switch statement to include new cases
3. Configure new clips in Inspector

Integration with Other Systems
- Compatible with state management systems
- Easy integration with global audio managers
- Support for custom stage change events

This system provides a robust and flexible solution for stage-based audio management in Unity applications, offering a smooth and customizable user experience.

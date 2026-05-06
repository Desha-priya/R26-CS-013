# Ransomware Killer

**Component Owner:** [Mate 2 Name]

## Description
Endpoint Behavioral Protection Layer.  
Monitors file system and process behavior to detect and stop ransomware encryption in real time.

## Key Responsibilities
- Real-time file entropy and process monitoring
- Rapid encryption detection
- Automatic process kill and file rollback

## Current Status
- Literature review on behavioral ransomware detection completed
- Dataset exploration in progress
- Basic monitoring logic under development

## Folder Contents
- `agent.py` → Lightweight endpoint agent
- `detector.py` → Core ransomware detection logic
- `rollback.py` → File recovery mechanisms

## Next Tasks
- Implement entropy-based detection
- Add process tree monitoring
- Develop rollback using snapshots
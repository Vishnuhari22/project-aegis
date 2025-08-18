# Project Aegis: Real-Time Anomaly Detection System

**Project Aegis is an end-to-end intelligent surveillance system designed to automatically detect and alert on anomalies in real-time video feeds.** This project goes beyond simple object detection to build a full-stack, scalable AI application that mirrors the challenges of production-grade systems.

This document covers the successful completion of Phase 1, which resulted in a fully functional local prototype capable of identifying an "unattended package" anomaly and broadcasting alerts to a live web dashboard.

## Demo

_(A GIF demonstrating the final prototype in action will be placed here)_

## Features Completed in Phase 1

*   **AI-Powered Vision Engine:** Implemented a core detection engine using Python, OpenCV, and a YOLOv8 model to identify and extract data for multiple object classes from video streams.
    
*   **Stateful Object Tracking:** Designed and built a custom, stateful centroid tracker from scratch. This algorithm assigns and maintains unique, persistent IDs for objects as they move through the frame, forming the foundation for all time-based analysis.
    
*   **Complex Anomaly Detection Logic:** Engineered the core anomaly detection rules to identify an "unattended package." This involved creating a state machine to manage object relationships (ownership), timers, and status changes (`owned`, `unattended`, `anomaly`).
    
*   **Robust Engineering & Debugging:** Systematically diagnosed and fixed complex bugs related to AI model "flicker," object occlusion, and state management, resulting in a more resilient and accurate prototype.
    
*   **Continuous Integration (CI):** Set up a professional CI pipeline with GitHub Actions and `flake8` to automatically lint and validate code quality on every commit.
    
*   **Web-Based Alerting:** Developed a web server using FastAPI and WebSockets to broadcast real-time anomaly alerts to a clean, simple HTML/JavaScript frontend.
    

## Technology Stack

*   **Programming Language:** Python
    
*   **Computer Vision:** OpenCV, Ultralytics (for YOLOv8)
    
*   **Web Backend:** FastAPI
    
*   **Real-time Communication:** WebSockets
    
*   **Server:** Uvicorn
    
*   **DevOps:** Git, GitHub Actions
    

## How to Run the Local Prototype

### 1\. Prerequisites

*   Python 3.8+
    
*   Git
    

### 2\. Setup

Clone the repository and set up the virtual environment.

    # Clone the repository
    git clone https://github.com/your-username/project-aegis.git
    cd project-aegis
    
    # Create and activate a virtual environment
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    
    # Install the required dependencies
    pip install -r requirements.txt
    

### 3\. Add Test Video

Place your test video file in the root of the project folder and ensure it is named `test_video1.mp4`.

### 4\. Run the Server

Launch the application using the Uvicorn server.

    uvicorn server:app --reload
    

The server will start on `http://127.0.0.1:8000`.

### 5\. View the Dashboard

*   The OpenCV video feed window will appear on your screen, showing the live detections and tracking.
    
*   Open a web browser and navigate to **`http://127.0.0.1:8000`** to see the live alert dashboard.
    

## Project Structure

    project-aegis/
    │
    ├── .github/
    │   └── workflows/
    │       └── ci.yml          # GitHub Actions CI workflow for linting
    │
    ├── aegis_core.py           # Core video processing and anomaly detection logic
    ├── server.py               # FastAPI web server and WebSocket endpoint
    ├── index.html              # Frontend HTML/JS for the alert dashboard
    ├── requirements.txt        # Project dependencies
    ├── test_video1.mp4         # Test video file
    └── README.md               # This file

# Control-System-Camera AI-Powered Productivity Assistant (CAPA)

CAPA is an intelligent productivity assistant built to support students and professionals in maintaining focus during study or work sessions. By leveraging standard webcam input and Python-based computer vision, the system analyzes desk activity, detects user presence, and identifies signs of distraction or sustained engagement.

## Key Features

- **Desk Presence Detection**: Detects whether a user is present and actively positioned at their desk.
- **Focus Monitoring**: Tracks motion patterns and activity to infer periods of concentration.
- **Distraction Alerts**: Identifies potential distraction behaviors and provides a foundation for feedback or logging.
- **Python-Driven Computer Vision**: Built on accessible Python libraries to make development and customization simple.

Enhanced features:

- **Head Detection (Frontal + Profile)**: Detects head position even when the user turns their head away from the camera, enabling more reliable presence detection.
- **Eye Recognition & Attention Assessment**: Detects eyes within head regions to determine if the user is looking at the screen, has closed eyes, or if eyes are unrecognizable — used to infer distracted or focused states.
- **Looking-Down / Orientation Detection**: Tracks vertical head position and profile orientation to flag when a user is looking down or away.
- **Security-Themed Live UI**: Real-time overlay with head/eye boxes, state labels, and a compact sidebar for quick session metrics and alerts.
- **Session Transcript & AI Summary**: Logs key events and uses a local LLM (optional) to generate a concise AI-powered session summary and recommendations.
- **Detailed Report Generation**: After each session, CAPA saves a comprehensive text report (`CAPA_Session_Report_YYYYMMDD_HHMMSS.txt`) including timeline, statistics, motion analysis, expressions, head orientation, and AI insights.

## Why CAPA

CAPA helps users build better study and work habits by providing an automated way to observe behavior in the workspace. It is especially useful for:

- Students preparing for exams or study sessions
- Remote workers seeking improved productivity
- Anyone who wants objective insight into desk-focused time

## Project Structure

- `CAPA.ipynb` — Interactive notebook for prototyping, testing, and demonstrating the camera-based productivity assistant.
- `README.md` — Project overview, installation guidelines, and usage instructions.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Webcam or compatible camera device
- Recommended Python packages: `opencv-python`, `numpy`, `matplotlib` (or similar CV dependencies)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/2320633/CAPA.git
cd CAPA
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install the core dependencies manually:
>
> ```bash
> pip install opencv-python numpy matplotlib requests
> ```

### Local LLM Setup (Optional)

CAPA can integrate with a local LLM for enhanced productivity suggestions and session summaries. To enable this feature:

1. **Install LM Studio** or another OpenAI-compatible LLM server
2. **Load a model**: Start your LLM server and load a model (recommended: `nvidia/nemotron-3-nano-4b` or similar)
3. **Verify API**: Ensure the server runs on `http://127.0.0.1:1234` with OpenAI-compatible endpoints
4. **Test connection**: The notebook will automatically test the API connection

> **Note**: If no model is loaded, CAPA will still function with basic camera monitoring but without AI-powered suggestions.

## Usage

1. Launch the notebook or script that initializes the camera and analytics pipeline.
2. Allow camera access when prompted.
3. Follow the on-screen output or notebook cells to view detection results and productivity insights.

### Start a Focus Session

Open `CAPA.ipynb` and run the cell that defines `run_focus_session_enhanced()` (or import the function into a script). Start a session from the notebook by running:

```python
session_events, transcript = run_focus_session_enhanced(min_motion_area=1500)
```

- Controls: Press `S` to start monitoring, `C` to change camera, `Q` to quit, or click the on-screen STOP area.
- The UI displays a live window with detected head/eye boxes, state labels, and a security-themed sidebar.

### Generated Detailed Report

After each session the notebook generates a comprehensive text report saved to the working directory. The report file is named like:

`CAPA_Session_Report_YYYYMMDD_HHMMSS.txt`

The report includes:
- Session overview (duration, events, alerts)
- Focus statistics (focused, distracted, absent percentages)
- Motion analysis (average, peak, minimum)
- Expression breakdown (NORMAL, EYES_CLOSED, LOOKING_DOWN, etc.)
- Head orientation analysis (frontal vs profile)
- AI-generated session analysis (if local LLM available)
- Recommendations and suggested improvements
- Detailed event timeline and session transcript

The notebook will print the filename and full path when the report is saved and will also preview the report contents in the notebook output.

## Customization

- Adjust detection thresholds, region-of-interest parameters, or behavior rules in the notebook or Python script.
- Extend the system to log session metrics, trigger notifications, or integrate with productivity tools.

## Contribution

Contributions are welcome. To propose enhancements or fixes:

1. Open an issue describing the feature or bug.
2. Fork the repository.
3. Submit a pull request with your improvements.

## License

This project is open source and may be used, modified, and shared under the terms of its chosen license.

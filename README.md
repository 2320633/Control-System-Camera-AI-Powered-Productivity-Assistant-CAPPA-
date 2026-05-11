# Control-System-Camera AI-Powered Productivity Assistant (CAPA)

CAPA is an intelligent productivity assistant built to support students and professionals in maintaining focus during study or work sessions. By leveraging standard webcam input and Python-based computer vision, the system analyzes desk activity, detects user presence, and identifies signs of distraction or sustained engagement.

## Key Features

- **Desk Presence Detection**: Detects whether a user is present and actively positioned at their desk.
- **Focus Monitoring**: Tracks motion patterns and activity to infer periods of concentration.
- **Distraction Alerts**: Identifies potential distraction behaviors and provides a foundation for feedback or logging.
- **Python-Driven Computer Vision**: Built on accessible Python libraries to make development and customization simple.

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
> pip install opencv-python numpy matplotlib
> ```

## Usage

1. Launch the notebook or script that initializes the camera and analytics pipeline.
2. Allow camera access when prompted.
3. Follow the on-screen output or notebook cells to view detection results and productivity insights.

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

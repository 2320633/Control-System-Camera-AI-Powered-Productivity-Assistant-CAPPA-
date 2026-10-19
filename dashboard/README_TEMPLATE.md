# {{TITLE}}

> {{TAGLINE}}

![CAPA repository dashboard](./dashboard/assets/repository-summary.svg)

{{OVERVIEW}}

**Project repository:** [{{REPOSITORY}}]({{REPOSITORY_URL}})

## Highlights

{{FEATURES}}

## Project snapshot

| Metric | Value |
| --- | ---: |
| Python source lines | {{PYTHON_LINES}} |
| Declared dependencies | {{DEPENDENCY_COUNT}} |
| Session report examples | {{REPORT_COUNT}} |
| Project files | {{PROJECT_FILES}} |
| Git commits | {{COMMIT_COUNT}} |
| Languages and formats | {{LANGUAGES}} |

The inventory excludes generated session report contents and local runtime state. Report examples are counted by filename only; their personal session data is not reproduced here.

## Progress and achievements

{{ACHIEVEMENTS}}

## Project structure

{{PROJECT_STRUCTURE}}

## Getting started

### Requirements

- Python 3.8 or later
- A webcam supported by OpenCV

### Install

```bash
git clone {{CLONE_URL}}
cd {{CLONE_DIRECTORY}}
python -m venv .venv
```

Activate the virtual environment, then install the project dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

### Run a session

Start the desktop monitor with:

```bash
python CAPA.py
```

Allow camera access when prompted. The OpenCV window supports **S** to begin, **C** to switch cameras, and **Q** to quit. The interactive walkthrough is also available in [CAPA.ipynb](./CAPA.ipynb).

### Optional local AI

For AI-generated prompts and summaries, start an OpenAI-compatible local model server at `http://127.0.0.1:1234` before running CAPA. The current implementation requests `nvidia/nemotron-3-nano-4b`; the local server must have a compatible model loaded. Camera monitoring and basic session analysis are separate from this optional service.

## Roadmap

The items below are generated from the current repository contents. A milestone disappears automatically when its corresponding tests, evaluation assets, calibration guidance, or test workflow are added.

{{ROADMAP}}

## Recent development

- **Latest commit:** {{LATEST_COMMIT}}
- **Latest commit date:** {{LATEST_COMMIT_DATE}}
- **Recent commits:**

{{RECENT_COMMITS}}

## Privacy and interpretation

CAPA reads webcam frames for on-device OpenCV analysis. The application writes text session reports to its working directory; inspect and manage those files as personal data. When enabled, AI prompts and session summaries are sent to the configured local model server. Focus and attention labels are heuristic signals, not objective measurements, and can be affected by lighting, camera placement, and individual differences.

## Dashboard maintenance

This README and its summary graphic are regenerated daily by [`.github/workflows/dashboard.yml`](./.github/workflows/dashboard.yml). The generator uses Python's standard library, reads repository metadata and source files, and updates these generated files only when their content changes.

**Dashboard last generated:** {{GENERATED_DATE}} (UTC)

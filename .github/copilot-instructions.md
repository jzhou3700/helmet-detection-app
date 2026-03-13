# Copilot Instructions for helmet-detection-app

## Project Overview

`helmet-detection-app` is a computer-vision application that detects whether people in images or video streams are wearing helmets. The project is in its initial setup phase.

## Repository Layout

```
.
├── README.md                  # Project overview
└── .github/
    └── copilot-instructions.md  # This file
```

As the project grows, the expected structure will include:
- `src/` or a package directory — application source code
- `models/` — pre-trained or fine-tuned detection model weights
- `data/` — sample images or dataset references
- `tests/` — unit and integration tests
- `requirements.txt` / `pyproject.toml` — Python dependencies
- `Dockerfile` — containerised runtime (if applicable)

## Technology Stack

The project is expected to use:
- **Python 3.10+** as the primary language
- **OpenCV** (`cv2`) for image/video I/O and preprocessing
- **PyTorch** or **TensorFlow/Keras** for the deep-learning model
- **YOLOv5/YOLOv8** or similar object-detection framework
- **pytest** for testing

## Build & Environment Setup

> Trust these instructions. Only search the repo if the information here is incomplete or found to be incorrect.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not yet exist, install the core stack manually:

```bash
pip install torch torchvision opencv-python ultralytics pytest
```

### 3. Run the application

```bash
python src/main.py --source <image_or_video_path>
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Lint

```bash
pip install ruff
ruff check .
```

## Key Conventions

- Follow **PEP 8** style. Use `ruff` for linting.
- Keep model weights out of version control (add to `.gitignore`).
- Place reusable utilities in a `utils/` sub-package.
- Write tests for every new module in `tests/` using `pytest`.
- Use `pathlib.Path` instead of `os.path` for file-system operations.
- Prefer relative imports within the package.

## CI / Validation

There are currently no GitHub Actions workflows. When adding one, the standard pipeline should:
1. Set up Python 3.10+
2. Install dependencies (`pip install -r requirements.txt`)
3. Run linting (`ruff check .`)
4. Run tests (`pytest tests/ -v`)

## Notes

- The repository is in early-stage setup; source files and tests do not yet exist.
- When adding new features, create the relevant source file under `src/`, add a corresponding test under `tests/`, and update `requirements.txt` as needed.

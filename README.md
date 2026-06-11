# Ocular-Cefálico-Silla-VR

**Smart Electric Wheelchair Control via Head and Ocular Movement in a Virtual Reality Environment**

[![Language](https://img.shields.io/badge/Unity-2021.3%20LTS-black.svg)](https://unity.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Headset](https://img.shields.io/badge/HMD-HTC%20Vive%20Pro%20Eye-orange.svg)](https://www.vive.com/)
[![Status](https://img.shields.io/badge/status-research-success.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

> Research code and Unity testbed accompanying the manuscript submitted to *Computers in Human Behavior* (Elsevier). The repository implements a full pipeline — VR data acquisition, elastic time-series similarity analysis, and ensemble classification — for hands-free wheelchair control based on head and gaze kinematics in immersive virtual environments.

---

## Table of Contents

1. [Citation](#citation)
2. [Overview](#overview)
3. [Scientific Contributions](#scientific-contributions)
4. [Repository Structure](#repository-structure)
5. [Hardware and Software Requirements](#hardware-and-software-requirements)
6. [Module 1 — VR Testbed (Unity)](#module-1--vr-testbed-unity)
7. [Module 2 — Elastic Similarity Measures](#module-2--elastic-similarity-measures)
8. [Module 3 — ML Classifiers and Operational Frameworks](#module-3--ml-classifiers-and-operational-frameworks)
9. [End-to-End Workflow](#end-to-end-workflow)
10. [Reproducibility](#reproducibility)
11. [Data Availability](#data-availability)
12. [License](#license)
13. [Authors and Affiliation](#authors-and-affiliation)
14. [Acknowledgments](#acknowledgments)
15. [Contact](#contact)

---

## Citation

If you use this repository, the testbed, the datasets, or any of the derived code in academic work, please cite both the companion article and the software artifact.

**Journal article (under review):**

```bibtex
@article{MitreHernandez2026OcularCefalico,
  author  = {Authors]},
  title   = {Elastic similarity ensembles and a hybrid MSM--LSTM classifier for head-motion command recognition: A multi-criteria framework applied to assistive wheelchair control in virtual reality},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

**Software / code artifact:**

```bibtex
@software{MitreHernandez2026OcularCefalicoCode,
  author  = {Autores},
  title   = {{Ocular-Cef{\'a}lico-Silla-VR}: Unity testbed, elastic similarity
             measures, and Proximity Forest 2.0 classifiers for VR-based
             wheelchair control},
  year    = {2026},
  url     = {https://github.com/HugoMitre/Ocular-Cefalico-Silla-VR},
  version = {1.0.0},
  institution = {Centro de Investigaci{\'o}n en Matem{\'a}ticas (CIMAT), Unidad Zacatecas}
}
```

> Replace the version tag and add a DOI once a Zenodo release is archived. We recommend creating a tagged release (`v1.0.0`) and minting a DOI through the GitHub–Zenodo integration for permanent citation.

---

## Overview

This repository contains the complete computational artifact behind a study on **hands-free electric wheelchair control** for users with severe upper-limb motor disability. The system records head and gaze kinematics inside an immersive VR supermarket scenario rendered on an **HTC Vive Pro Eye** headset, characterizes the resulting trajectories using **elastic time-series similarity measures**, and classifies them into **eight directional wheelchair commands** through two complementary models:

- a **Proximity Forest 2.0** variant tailored to bivariate angular signals, and
- a **hybrid MSM–LSTM pipeline** with confidence-based late fusion.

The work explicitly reframes inter-subject behavioral variability in head motion as **intrinsic signal**, not noise, and treats systematic confusions in diagonal commands as **HCI design constraints**, aligning the methodology with the editorial scope of *Computers in Human Behavior*.

---

## Scientific Contributions

| # | Contribution | Where in the repository |
|---|---|---|
| C1 | A Unity-based VR testbed integrated with the HTC Vive Pro Eye for synchronized head/gaze acquisition, contextual auditory guidance, and reference-route deviation logging. | [`realidad virtual/`](realidad%20virtual) |
| C2 | Four reproducible **behavioral head-motion metrics**: directional permanence, angular standard deviation, angular velocity, and dwell time. | [`realidad virtual/HeadMovementData/`](realidad%20virtual/HeadMovementData), [`realidad virtual/Route/`](realidad%20virtual/Route) |
| C3 | A unified, FLOP-instrumented implementation of **nine to ten elastic similarity measures** (DTW, WDTW, cDTW, ADTW, ERP, EDR, LCSS, TWE, MSM, SSDTW, ESDTW) with shared visualization and benchmarking infrastructure. | [`Medidas de distancia elástica/`](Medidas%20de%20distancia%20el%C3%A1stica) |
| C4 | A **multi-criteria comparator** that ranks similarity measures via the trade-off `F(x) = α·Cost_comp(x) − β·Accuracy_align(x)`, enabling principled algorithm selection for embedded VR pipelines. | [`Medidas de distancia elástica/ Measurement Evaluation & Trade-off Ranking Interactive Comparator/`](Medidas%20de%20distancia%20el%C3%A1stica/%20Measurement%20Evaluation%20%26%20Trade-off%20Ranking%20Interactive%20Comparator) |
| C5 | A **Proximity Forest 2.0** classifier specialized for head-angle time series, plus three companion classifiers for velocity, navigational deviation, and dwell time. | [`ML-Classifiers/`](ML-Classifiers) |
| C6 | A **real-time operational layer** exposing the trained models as TCP/JSON microservices (ports 5556–5559) consumable by the Unity client. | [`ML-Classifiers/*_operational_framework.py`](ML-Classifiers) |
| C7 | A **synthetic trajectory generator** with adaptive and realistic kinematics for pre-trial validation and ground-truth construction. | [`realidad virtual/Recopilación de datos/Synthetic Trajectories with Adaptive and Realistic Kinematics.py`](realidad%20virtual/Recopilaci%C3%B3n%20de%20datos) |

---

## Repository Structure

```
Ocular-Cefalico-Silla-VR/
├── realidad virtual/                  # Unity VR testbed (C#)
│   ├── Control/                       # Keyboard / Arduino-STM32 joystick / WebSocket bridge
│   ├── Environment-Optimization/      # Mesh analyzers, decimators, GPU instancing
│   ├── HeadMovementData/              # Head direction, permanence, σ, velocity
│   ├── OcularMovementData/            # Gaze direction, permanence, σ, velocity
│   ├── Route/                         # Bezier reference paths, deviation, dwell tracking
│   ├── Instructions/                  # Stage-based contextual audio manager
│   └── Recopilación de datos/         # Multi-stream CSV logger + cybersickness recorder
│                                      # + synthetic trajectory generator
│
├── Medidas de distancia elástica/     # 10+ elastic similarity measures (Python)
│   ├── Dynamic Time Warping (DTW).py
│   ├── Weighted Dynamic Time Warping (WDTW).py
│   ├── Constrained Dynamic Time Warping (CDTW).py
│   ├── Amerced/Adaptive DTW (ADTW).py
│   ├── Block Dynamic Alignment (BDTW).py
│   ├── Edit Distance with Real Penalty (ERP).py
│   ├── Longest Common Subsequence (LCSS).py
│   ├── Time Warp Edit (TWE).py
│   ├── Move-Split-Merge (MSM).py
│   ├── Shape Segment DTW (SSDTW).py
│   ├── Extrema-based Shape DTW (ESDTW).py
│   ├── Ground Truth/                  # Ground-truth generator and data replicators
│   └── Measurement Evaluation & Trade-off Ranking Interactive Comparator/
│
└── ML-Classifiers/                    # Proximity Forest 2.0 framework (Python)
    ├── wheelchair_command_classifier.py
    ├── velocity_classifier.py
    ├── deviation_classifier.py
    ├── dwell_classifier.py
    ├── command_operational_framework.py
    ├── velocity_operational_framework.py
    ├── Deviation_operational_framework.py
    └── dwell_operational_framework.py
```

---

## Hardware and Software Requirements

**Virtual reality hardware**

- HTC Vive Pro Eye head-mounted display with integrated Tobii eye tracker
- SteamVR-compatible host workstation (Windows 10/11, NVIDIA RTX-class GPU recommended)
- Optional: STM32-based analog joystick (used for baseline manual control)

**Unity environment**

- Unity 2021.3 LTS (or higher)
- SRanipal SDK (eye tracking)
- SteamVR Plugin
- UnityMeshSimplifier (`https://github.com/Whinarn/UnityMeshSimplifier.git`)

**Python environment**

- Python ≥ 3.8 (3.10 recommended)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`, `scikit-learn`, `joblib`
- For the LSTM branch of the hybrid pipeline: `tensorflow` ≥ 2.10 or `pytorch` ≥ 1.13
- Real-time interface: standard library `socket` and `json` (no extra dependencies)

A consolidated `requirements.txt` can be regenerated from the imports of each module; we recommend an isolated virtual environment per submodule.

---

## Module 1 — VR Testbed (Unity)

The Unity project under [`realidad virtual/`](realidad%20virtual) implements the experimental environment used for data acquisition. It is organized along the participant pipeline:

1. **`Control/`** — three input modalities are supported in parallel for comparative studies: keyboard (`KeyboardMovementController.cs`), physical STM32 joystick streamed over a WebSocket bridge (`WheelchairjoystickController.cs`, `joystick_bridge.py`, `JoystickSignalConditioner.ino`), and an automatic path follower for benchmarking (`AutoPathFollower.cs`).
2. **`Environment-Optimization/`** — mesh complexity analysis, batch GPU instancing, and progressive mesh decimation to sustain a stable ≥ 90 Hz frame rate on the Vive Pro Eye.
3. **`HeadMovementData/`** — derives the four behavioral metrics (direction, permanence, angular standard deviation, angular velocity) from `Camera.main` rotation relative to a configurable frontal reference.
4. **`OcularMovementData/`** — analogous metrics for gaze, driven by `LineRenderer` rays from the Tobii eye tracker.
5. **`Route/`** — Bezier reference trajectories (`CustomCurve.cs`), continuous deviation measurement (`PathPerformanceTracker.cs`), contextual zone triggers, and dwell tracking on stops.
6. **`Instructions/`** — stage-based auditory cues (`EtapaAudioManager`) and ambient supermarket atmosphere.
7. **`Recopilación de datos/`** — multi-stream logger (`BehavioralDataLogger.cs`, `DataCombiner.cs`), `ParticipantSelector` (Editor folder, with PlayerPrefs persistence), `CybersicknessRecorder.cs` (Spacebar toggle), and the synthetic trajectory generator used for ground truth.

All logs are written to CSV with per-participant folders. **Output paths are hard-coded** in the legacy scripts (`C:\Users\Manuel Delgado\Documents\...`) and should be customized before deployment.

---

## Module 2 — Elastic Similarity Measures

The [`Medidas de distancia elástica/`](Medidas%20de%20distancia%20el%C3%A1stica) package provides reference Python implementations of the elastic similarity measures evaluated in the paper. Each script is self-contained and follows a common contract:

- bivariate input (X/Y angular series),
- exact FLOP counting through NumPy operator interception,
- cost-matrix and warping-path visualization,
- per-pair similarity reporting in CSV/Excel.

**Algorithms implemented**

| Family | Algorithm | Distinctive parameter |
|---|---|---|
| Classical | DTW | — |
| Weighted | WDTW | sigmoid slope `g` |
| Constrained | cDTW, ICDTW | Sakoe–Chiba band `r` |
| Adaptive | ADTW, BDTW | adaptive band, block size |
| Edit-based | ERP, EDR | gap penalty `g`, tolerance `ε` |
| Subsequence | LCSS | tolerance `ε`, lag `δ` |
| Edit + warp | TWE | stiffness `ν`, penalty `λ` |
| Three-op | MSM | move–split–merge cost `c` |
| Shape | SSDTW, ESDTW | Haar levels `L`, shape weights |

**Multi-criteria comparator.** The [`Measurement Evaluation & Trade-off Ranking Interactive Comparator`](Medidas%20de%20distancia%20el%C3%A1stica/%20Measurement%20Evaluation%20%26%20Trade-off%20Ranking%20Interactive%20Comparator) module aggregates the per-algorithm results and ranks them according to

```
F(x) = α · Cost_comp(x) − β · Accuracy_align(x)
```

with interactive selection of `α` and `β`. Three canonical configurations are pre-defined: *efficiency-focused* (α = 0.8, β = 0.2), *precision-focused* (α = 0.2, β = 0.8), and *balanced* (α = β = 0.5). Reports are exported as multi-sheet Excel workbooks including normalized metrics, mathematical breakdowns, and comparative plots.

**Ground truth.** The [`Ground Truth/`](Medidas%20de%20distancia%20el%C3%A1stica/Ground%20Truth) directory provides a deterministic generator and adaptive path transformers used to build the reference alignments against which the elastic measures are scored.

---

## Module 3 — ML Classifiers and Operational Frameworks

The [`ML-Classifiers/`](ML-Classifiers) package operationalizes the empirical findings of Module 2 into four production-grade classifiers, all built on a **Proximity Forest 2.0** backbone with mixed elastic measures and an information-gain splitting criterion.

| Classifier | Input features | Output space | Backbone |
|---|---|---|---|
| `wheelchair_command_classifier` | `HeadAngleX`, `HeadAngleY` | 8 directional commands (`front`, `back`, `left_turn`, `right_turn`, and the four diagonals) | 150 trees, depth 16, MSM (0.5) + SSDTW (0.3) + ERP (0.2) |
| `velocity_classifier` | `HeadVelocityX`, `HeadVelocityY` | 7 magnitude levels (`MUY_BAJA` → `MUY_ALTA`) | 100 trees, depth 15 |
| `deviation_classifier` | `HeadDeviationX`, `HeadDeviationY` | 5 navigation patterns (`NAVEGACION_EFICIENTE`, `NAVEGACION_DIRIGIDA`, `EXPLORACION_PAUSADA`, `EXPLORACION_ACTIVA`, `BUSQUEDA_REORIENTACION`) | 100 trees, depth 12 |
| `dwell_classifier` | `HeadAngleX`, `HeadAngleY`, dwell duration | 5 fixation patterns | 20 trees, depth 12 |

Each classifier is paired with an **operational framework** (`*_operational_framework.py`) that:

- auto-loads `proximity_forest_model.pkl`, `command_mapping.pkl`, `schema.pkl`, and `training_metadata.pkl`,
- exposes a **TCP/JSON server** on ports 5556–5559 for Unity-side consumption,
- supports both **synthetic validation mode** (with embedded ground truth) and **production mode** (real-time inference), and
- emits per-call latency and accuracy telemetry.

**Communication protocol (excerpt)**

```jsonc
// Request (Unity → Python)
{
  "HeadAngleX":    [/* float array */],
  "HeadAngleY":    [/* float array */],
  "ground_truth":  "front"                 // optional, synthetic mode only
}

// Response
{
  "prediction": "front",
  "confidence": 0.95,
  "time_ms":    45.2,
  "correct":    true
}
```

Reported test-time performance on synthetic data is 100% (command), ≥ 95% (velocity), ≥ 92% (deviation), and ≥ 88% (dwell), with average inference latency below 50 ms per request.

---

## End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HTC Vive Pro Eye + Unity 2021                      │
│      (head/gaze acquisition, contextual audio, Bezier route)         │
└──────────────────────────────────────────────────────────────────────┘
                              │  CSV (per participant)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│           Elastic similarity analysis (Module 2)                      │
│   DTW · WDTW · cDTW · ADTW · ERP · EDR · LCSS · TWE · MSM · SSDTW    │
│   + multi-criteria F(x) = α·Cost − β·Accuracy ranking                 │
└──────────────────────────────────────────────────────────────────────┘
                              │  selected metric(s)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│   Proximity Forest 2.0 training (Module 3) → *.pkl models             │
└──────────────────────────────────────────────────────────────────────┘
                              │  PKL bundle
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│   TCP/JSON operational servers (5556–5559) ⇄ Unity client             │
│                       8 wheelchair commands                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Reproducibility

```bash
# 1. Clone the repository
git clone https://github.com/HugoMitre/Ocular-Cefalico-Silla-VR.git
cd Ocular-Cefalico-Silla-VR

# 2. (Optional) create an isolated environment
python -m venv .venv && source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate                              # Windows

# 3. Install Python dependencies
pip install numpy pandas scipy matplotlib openpyxl scikit-learn joblib

# 4. Run a representative elastic measure on the sample data
python "Medidas de distancia elástica/Move-Split-Merge (MSM).py"

# 5. Train the directional-command classifier
python "ML-Classifiers/wheelchair_command_classifier.py"

# 6. Launch the operational TCP server
python "ML-Classifiers/command_operational_framework.py"

# 7. Open the Unity project under realidad virtual/ in Unity 2021.3 LTS
#    and run the main scene.
```

A deterministic seed is set inside each training script; runs are expected to reproduce within ±0.5% accuracy on the synthetic test split.

---

## Data Availability

Raw participant recordings are governed by the ethics protocol of the parent study and are available from the corresponding author on reasonable request, subject to a data-use agreement. The repository ships with **synthetic trajectory generators** (`realidad virtual/Recopilación de datos/Synthetic Trajectories with Adaptive and Realistic Kinematics.py` and `Medidas de distancia elástica/Ground Truth/`) that reproduce the kinematic envelope of the real data and allow end-to-end testing of the pipeline without access to the human-subject corpus.

---

## License

Released under the **MIT License** (see `LICENSE` file). Third-party Unity assets and SDKs (SRanipal, SteamVR Plugin, UnityMeshSimplifier) retain their respective licenses.

---

## Authors and Affiliation

- **Hugo Mitre-Hernández** — Centro de Investigación en Matemáticas (CIMAT), Unidad Zacatecas, México. Research areas: human–computer interaction, machine learning, assistive technology.
- *Co-authors as listed in the journal manuscript.*

---

## Acknowledgments

This work was developed at **CIMAT Zacatecas** within a research line on intelligent assistive technologies and immersive HCI. The authors thank the participants of the VR study and the technical staff supporting the eye-tracking infrastructure.

---

## Contact

For scientific correspondence, replication requests, or questions regarding the testbed, please open an issue on the [GitHub repository](https://github.com/HugoMitre/Ocular-Cefalico-Silla-VR/issues) or contact the corresponding author through the affiliation above.

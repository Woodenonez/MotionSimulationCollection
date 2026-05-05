# MotionSimulationCollection

Synthetic motion-simulation dataset generation and visualization tools for trajectory prediction experiments.

This is quite old and only used for my previous research work. 

## What this repo includes

- Multi-scene dataset generators (hospital, warehouse, assembly, multimodal scenes)
- Single-interaction scenario generators
- Data assembly helpers and CSV aggregation utilities
- Quick visualization and dataset sanity-check scripts

Main entry scripts live in `src/`:

- `gen_ald_dataset.py`
- `gen_bsd_dataset.py`
- `gen_hpd_dataset.py`
- `gen_msmd_dataset.py`
- `gen_sid_dataset.py`
- `gen_sidv2_dataset.py`
- `gen_wsd_dataset.py`
- `viz_all_data.py`
- `test_dataset.py`

## Requirements

- Python 3.11+
- Dependencies from `pyproject.toml` (numpy, pandas, matplotlib, torch, torchvision, etc.)

## Setup

```bash
# from repository root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick usage

Run from the repository root:

```bash
python src/gen_ald_dataset.py
python src/gen_hpd_dataset.py
python src/gen_sid_dataset.py
python src/gen_wsd_dataset.py
```

Visualize generated trajectories:

```bash
python src/viz_all_data.py
```

Quick dataset loading/sanity check:

```bash
python src/test_dataset.py
```

Generated outputs are written under folders such as `Data/...` or `Data_V1/...` based on the selected generator script.
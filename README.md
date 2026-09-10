# Tibetan Plateau DSSR S2S

Code for subseasonal prediction of downward surface shortwave radiation (DSSR) over the Tibetan Plateau using a deterministic U-Net and residual diffusion ensemble.

The workflow includes:

- DSSR daily aggregation and seven-day history construction
- ECMWF S2S GRIB/NetCDF preprocessing and ensemble statistics
- deterministic U-Net forecasting
- conditional residual diffusion with DDIM sampling
- ensemble recentering and forecast verification

## Structure

`src/data` prepares DSSR and S2S inputs. `src/models` and `src/diffusion` contain the forecasting models. `src/evaluation` contains deterministic and ensemble metrics. `configs` holds the data, model and final-run settings.

## Data

Input data, terrain arrays and solar-geometry arrays are not included. Set their locations in `configs/paths.yaml`.

## Usage

```bash
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
```

The released final configuration uses 8-step DDIM sampling, 30 ensemble members, recentering and residual scale `0.15`.

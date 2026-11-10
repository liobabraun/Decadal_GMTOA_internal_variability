# Decadal_GMTOA_internal_variability

Master Thesis "The Role of Pacific Decadal and Atlantic Multidecadal Variability in Modulating Top-of-Atmosphere Net Radiation - A Storyline Approach" in M.Sc. Earth System Data Science and Remote Sensing at Leipzig University. The full thesis is available upon request from the author.

## Analysis notebooks

The numbered notebooks form the main analysis pipeline and should be run in
order:

- **`01_piControl_annual_anomalies`**  
  Calculates annual anomalies for all variables and derives the annual PDV and
  AMV indices. This is the only notebook that reads the raw model output.
  It writes `<name>_annual_anom.nc` and `PDV_AMV_annual.nc`.

- **`02_piControl_decadal_storylines`**  
  Classifies 110 non-overlapping decades into four PDV/AMV storylines and
  calculates composite maps for the variables. It reads the output of notebook
  01 and writes `piControl/ivs_SST.nc` and `piControl/decadal/`.

- **`03_piControl_lowpass_MLR`**  
  Applies a 10-year Lanczos low-pass filter and performs multiple linear
  regression of GMTOA on PDV and AMV states and rates of change. It reads the
  output of notebooks 01 and 02 and writes
  `piControl/lanczos/mlr_coef_maps.nc`.

- **`04_piControl_state_transition`**  
  Performs lead-lag regressions of TOA on PDV and AMV and analyses states,
  transitions, and the contributions of SW, LW, clouds, and LTS. It reads the
  output of notebook 01 and writes results to `piControl/lanczos/`.

## Supplementary notebook

The notebook `derive_omega500_regions` derives the mean vertical pressure
velocity at 500 hPa and documents the three analysis regions used in the study.
It writes
`data/piControl/wap_500_mean.nc` and displays the `omega500_regions` map; it is
not required to run notebooks 01–04.

## Shared functions

The file `functions.py` contains helper functions used by the notebooks for
data loading and preprocessing, anomaly calculation, region selection,
storyline classification, filtering, regression, significance testing, and
plotting. It also stores the common definitions of the three analysis regions
used throughout the analysis.

All anomalies are annual means with the first 100 years dropped (spin-up) and the linear trend removed, which leaves 1100 years. 03 and 04 then apply the low-pass filter, which leaves 1060 years.

## Data

The input is the CESM2 piControl run of CMIP6 (`r1i1p1f1`, 1200 years, monthly). It can be downloaded from ESGF (e.g. https://metagrid.esgf-west.org/search, source `CESM2`, experiment `piControl`).

| variable | table | used for |
|---|---|---|
| `ts` | Amon | SST, PDV and AMV indices |
| `sftlf` | fx | land-sea mask |
| `rsdt`, `rsut`, `rlut` | Amon | TOA shortwave and longwave |
| `ta`, `tas` | Amon | lower-tropospheric stability |
| `cl` | Amon | low, middle and high cloud cover (and `ps` for the pressure levels) |
| `wap` | Amon | vertical pressure velocity at 500 hPa and definition of the analysis regions |

01 expects one folder per variable (`data_dir/ts/*.nc`, `data_dir/cl/*.nc`, …). The files of a folder are read together and do not have to be merged first.

## Running

1. Set the paths in the **Config** cell at the top of each notebook: `data_dir` and `mask_path` for the raw data (01), and the data and output folders (`anom_dir`, `out_dir`, `results_dir`, `decadal_dir`). The supplementary `derive_omega500_regions` notebook additionally needs the raw `wap` directory.
2. Run the numbered notebooks in order. Results that already exist are read back instead of being computed again; `overwrite = True` (01, 02) or `recompute = True` (03, 04) builds them anew. The supplementary notebook follows the same pattern: it loads `wap_500_mean.nc` if present and recalculates it only when `overwrite = True`.

The other settings are in the Config cell as well, e.g. the filter cutoff, the size of the neutral class, the significance level, the colour scales, and `save_figures` to write the figures to disk.
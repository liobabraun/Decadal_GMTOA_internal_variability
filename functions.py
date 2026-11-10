"""Shared utilities for the CESM2 storyline notebooks.

Used by both ``02_piControl_decadal_storylines.ipynb`` (decades of a long control run) and
``storylines_LE.ipynb`` / the per-variable LE notebooks (members of the CESM2
Large Ensemble).  The two setups differ only in how the sample dimension is
built:

    piControl : sample = 10-year blocks of the time axis, anomalies from
                ``clip_detrend_anom`` (drop spin-up, remove linear trend)
    LE        : sample = ensemble members over a fixed decade, anomalies from
                ``ensemble_anomalies`` (remove the ensemble mean = forced response)

Everything downstream of that -- indices, classification, significance testing,
compositing -- is shared.

Convention: **PDV on the x-axis, AMV on the y-axis**, with storyline classes

    0 = Neutral      1 = PDV+/AMV+    2 = PDV-/AMV+    3 = PDV-/AMV-    4 = PDV+/AMV-

The classification is produced in exactly one notebook and saved; every other
notebook reads it back instead of reclassifying.  Derived data is written to::

    /home/data/<experiment>/ivs_<name>.nc      storyline index per decade
    /home/data/<experiment>/decadal/<var>.nc   decadal field + index
    /home/figures/<experiment>/<name>.png      figures, only on request

where <experiment> is 'piControl' or 'LE'.  Figures are never written unless a
``save_path`` is passed explicitly.
"""

import pathlib
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection
from xarray.coding.cftime_offsets import MonthBegin


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Index 0 is the neutral class, 1-4 are the storylines.  Defined once here so
# that the classification plot, the composite maps and any boxplot in the
# per-variable notebooks can never drift apart.
IVS_labels = ['Neutral', 'PDV+/AMV+', 'PDV-/AMV+', 'PDV-/AMV-', 'PDV+/AMV-']
#IVS_colors = ['lightgray', 'tab:orange', 'tab:blue', 'tab:green', 'tab:red']
#IVS_colors = ['#BBBBBB', '#E69F00', '#0072B2', '#009E73', '#CC79A7']
IVS_colors = ['#BBBBBB', '#d96f02', '#7570b3', '#66a61e', '#e7298a']

# Storyline order used for boxplots / grouped figures (neutral excluded).
IVS_order = IVS_labels[1:]

# Analysis boxes, in the native 0..360 longitude convention of CESM2.
regions: Dict[str, Dict[str, Tuple[float, float]]] = {
    "southeastern_pacific":  {"lat": (-40, -15), "lon": (255, 290)},   # (-40, -20)?
    "northeastern_pacific": {"lat": (20, 40),   "lon": (220, 245)},   # (25, 40) ?
    "western_pacific":  {"lat": (-5, 10),   "lon": (105, 155)},   
}

# member label itransform_first=True,n CESM2-LE filenames, e.g. "...LE2-1301.020.cam.h0..."
_member_re = re.compile(r"LE2-(\d+\.\d+)")


# ---------------------------------------------------------------------------
# where derived data lives
# ---------------------------------------------------------------------------

# Everything this project writes goes under data_root/<experiment>/, so that
# piControl and LE results can never overwrite each other.
data_root = pathlib.Path('/Users/liobabraun/Desktop/Uni_Leipzig/MA/server/data')


def index_path(name: str = 'SST', experiment: str = 'piControl') -> pathlib.Path:
    """Path of a storyline index, e.g. /home/lbraun/data/piControl/ivs_SST.nc."""
    return data_root / experiment / f'ivs_{name}.nc'


def decadal_path(variable: str, experiment: str = 'piControl') -> pathlib.Path:
    """Path of a prepared decadal field, e.g. .../piControl/decadal/LTS.nc."""
    return data_root / experiment / 'decadal' / f'{variable}.nc'


# Figures are kept apart from the data, split by experiment the same way.
figure_root = pathlib.Path('/Users/liobabraun/Desktop/Uni_Leipzig/MA/server/figures')


def figure_path(name: str, experiment: str = 'piControl',
                ext: str = 'png') -> pathlib.Path:
    """Path of a saved figure, e.g. /home/lbraun/figures/piControl/LTS.png."""
    return figure_root / experiment / f'{name}.{ext}'


# ---------------------------------------------------------------------------
# loading and preprocessing
# ---------------------------------------------------------------------------

def preprocess(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
    """Standard preprocessing for a single CESM2-LE file.

    Tags the dataset with the member label taken from the filename and
    optionally drops everything except ``variables``.  Meant to be handed to
    ``xr.open_mfdataset(..., preprocess=...)``, e.g.

        lambda ds: preprocess(ds, variables=["FSNT"])
    """
    src = ds.encoding.get("source", "")
    member = _member_re.search(pathlib.Path(src).name).group(1)

    if variables is not None:
        ds_reduced = ds[variables].reset_coords(drop=True)
    else:
        ds_reduced = ds.reset_coords(drop=True)

    return ds_reduced.assign_coords(member=member)


def load_le_members(
    dir_path: str,
    variable: str,
    shift_month: bool = True,
    chunks: Optional[dict] = None,
    **open_kwargs,
) -> xr.Dataset:
    """Open all LE files for one variable and concatenate them along ``member``.

    This is the block that every per-variable notebook repeats, so it lives
    here instead.  ``dir_path`` is the parent directory holding one
    subdirectory per variable.

    ``shift_month`` corrects the CAM h0 time stamp: monthly means are labelled
    with the start of the *following* month, so without the shift the record
    runs 2015-02 .. 2025-01 and ``resample('YE')`` would produce 11 partial
    years instead of 10 complete ones.

    ``chunks`` defaults to ``{"time": -1}``.
    """
    var_dir = pathlib.Path(dir_path) / variable
    files = sorted(str(p) for p in var_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"no .nc files found in {var_dir}")

    ds = xr.open_mfdataset(
        files,
        preprocess=lambda d: preprocess(d, variables=[variable]),
        combine="nested",
        concat_dim="member",
        chunks={"time": -1} if chunks is None else chunks,
        **open_kwargs,
    )

    if shift_month:
        ds = ds.assign_coords(time=ds.time - MonthBegin(1))

    return ds


def ocean_mask(path: str, threshold: Optional[float] = None) -> xr.DataArray:
    """Build a boolean ocean mask (True = ocean) from a land-fraction file.

    Understands both conventions used here: ``LANDFRAC`` from CESM2-LE history
    files (fraction, 0..1, default cutoff 0.5) and ``sftlf`` from the CMIP-style
    piControl fixed fields (percent, 0..100, default cutoff 50).  A time
    dimension is dropped -- land fraction is constant -- so that a later
    ``.where(mask)`` never has to align along time.
    """
    ds = xr.open_dataset(path)

    if 'LANDFRAC' in ds:
        landfrac, default = ds['LANDFRAC'], 0.5
    elif 'sftlf' in ds:
        landfrac, default = ds['sftlf'], 50.0
    else:
        raise KeyError(f"neither 'LANDFRAC' nor 'sftlf' found in {path}")

    mask = landfrac < (default if threshold is None else threshold)
    if 'time' in mask.dims:
        mask = mask.isel(time=0, drop=True)
    return mask


def select_box(
    da: xr.DataArray,
    region: Optional[str] = None,
    lat: Optional[Tuple[float, float]] = None,
    lon: Optional[Tuple[float, float]] = None,
    coarsen: Optional[int] = None,
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Cut out an analysis box, optionally masking and coarsening the grid.

    Either pass a key from ``regions`` or explicit ``lat``/``lon`` bounds.
    ``coarsen=2`` averages 2x2 grid cells, which is what the scatter-plot
    sections of the per-variable notebooks do before ravelling.

    ``mask`` is applied *before* coarsening, so a 2x2 block that straddles a
    coastline averages only its ocean cells instead of smearing land into them.
    The block then stands for less area than a full one, which is why the box
    mean has to be weighted -- see ``box_weights``.
    """
    if region is not None:
        lat, lon = regions[region]['lat'], regions[region]['lon']
    if lat is None or lon is None:
        raise ValueError("pass either region=... or both lat=... and lon=...")

    out = da.sel(lat=slice(*lat), lon=slice(*lon))
    if mask is not None:
        out = out.where(mask.sel(lat=slice(*lat), lon=slice(*lon)))
    if coarsen:
        out = out.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()
    return out


def box_weights(coords, region: Optional[str] = None,
                lat: Optional[Tuple[float, float]] = None,
                lon: Optional[Tuple[float, float]] = None,
                coarsen: Optional[int] = None,
                mask: Optional[xr.DataArray] = None,
                min_ocean: float = 0.0) -> xr.DataArray:
    """Area weight of every (coarsened) cell of a box: cos(lat) x ocean fraction.

    Two corrections in one number, because they multiply:

    - **cos(lat)** - a grid cell at 40S covers noticeably less area than one at
      10S, so an unweighted box mean over a 30-degree band tilts towards the
      poleward edge.
    - **ocean fraction** - with a ``mask``, a coarsened block that is half land
      stands for half the ocean area of a full block and should count half as
      much. Without a mask this factor is 1 everywhere and only cos(lat) acts.

    ``coords`` is any object carrying the ``lat``/``lon`` of the data, normally
    the dataset itself. Weights are built on *its* coordinates so that they line
    up with ``select_box`` cell for cell.

    ``min_ocean`` drops blocks below that ocean fraction outright. The default
    keeps every block with any ocean in it and lets the weight do the work,
    which is the unbiased choice; raise it to exclude near-coastal cells whose
    mean rests on one or two points.
    """
    if region is not None:
        lat, lon = regions[region]['lat'], regions[region]['lon']
    if lat is None or lon is None:
        raise ValueError("pass either region=... or both lat=... and lon=...")

    lat_sel = coords['lat'].sel(lat=slice(*lat))
    lon_sel = coords['lon'].sel(lon=slice(*lon))

    if mask is None:
        ocean = xr.DataArray(
            np.ones((lat_sel.size, lon_sel.size)),
            dims=['lat', 'lon'], coords={'lat': lat_sel, 'lon': lon_sel},
        )
    else:
        ocean = mask.sel(lat=slice(*lat), lon=slice(*lon)).astype(float)

    if coarsen:
        ocean = ocean.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    weight = area_weights(ocean['lat']) * ocean
    if min_ocean:
        weight = weight.where(ocean >= min_ocean)
    return weight.rename('weight')


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------

def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """Cosine-of-latitude weights for area-weighted spatial means."""
    w = np.cos(np.deg2rad(lat))
    w.name = "weights"
    return w


def clip_detrend_anom(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """piControl anomalies: drop the first 100 years, then remove a linear trend.

    Returns anomalies with the same coordinates and dimensions as the input.
    """
    da = da.isel(**{time_dim: slice(100, None)})

    if da.chunks is not None:
        da = da.chunk({time_dim: -1})

    coeffs = da.polyfit(dim=time_dim, deg=1)
    trend = xr.polyval(da[time_dim], coeffs.polyfit_coefficients)
    da_detrended = da - trend

    return da_detrended - da_detrended.mean(dim=time_dim)


def ensemble_anomalies(obj, dim: str = "member"):
    """LE anomalies: subtract the ensemble mean.

    Works on a DataArray or a Dataset.
    """
    return obj - obj.mean(dim=dim)


def annual_anomalies_le(ds, mask: Optional[xr.DataArray] = None, dim: str = "member"):
    """Monthly LE data -> annual means -> anomalies about the ensemble mean.

    Convenience wrapper for the three lines every per-variable notebook needs.
    """
    if mask is not None:
        ds = ds.where(mask)
    annual = ds.resample(time='YE').mean()
    return ensemble_anomalies(annual, dim=dim)


def annual_anomalies_picontrol(da, mask: Optional[xr.DataArray] = None, freq: str = 'YS'):
    """Monthly piControl data -> annual means -> clipped, detrended anomalies.

    The piControl counterpart of ``annual_anomalies_le``.
    """
    annual = da.resample(time=freq).mean()
    if mask is not None:
        annual = annual.where(mask)
    return clip_detrend_anom(annual)


def transition(da, time_dim: str = "time"):
    """Rate of change per time step, robust to cftime axes.

    ``differentiate`` cannot handle cftime directly, so the time coordinate is
    temporarily replaced by an integer index and restored afterwards.  Applied
    to annual anomalies this gives the year-to-year SST transition.
    """
    real_t = da[time_dim]
    di = da.assign_coords({time_dim: np.arange(da.sizes[time_dim])}).differentiate(time_dim)
    return di.assign_coords({time_dim: real_t})


# ---------------------------------------------------------------------------
# derived variables
# ---------------------------------------------------------------------------

# R_d / c_p for dry air
kappa = 0.286


def potential_temperature(temperature: xr.DataArray, pressure_pa,
                          p0_pa: float = 100000.0) -> xr.DataArray:
    """Potential temperature from temperature [K] and pressure in Pa.

    ``pressure_pa`` is a single level (float) or a field such as the surface
    pressure ``ps`` (DataArray on the same grid as ``temperature``).
    """
    return temperature * (p0_pa / pressure_pa) ** kappa


def lower_tropospheric_stability(ta: xr.DataArray, tas: xr.DataArray,
                                 ps: xr.DataArray, plev_dim: str = 'plev',
                                 upper_pa: float = 70000.0,
                                 p0_pa: float = 100000.0) -> xr.DataArray:
    """LTS = theta(700 hPa) - theta(surface), after Klein & Hartmann (1993).

    theta(surface) = tas * (p0/ps)**kappa with the actual surface pressure ``ps``
    in Pa. 
    """
    levels = np.asarray(ta[plev_dim].values, dtype=float)

    # a Pa axis reaches ~1e5, an hPa axis stays below ~1050
    scale = 100.0 if levels.max() < 2000.0 else 1.0     # coordinate value -> Pa
    upper = upper_pa / scale
    if upper not in levels:
        raise ValueError(f"level {upper} not in {plev_dim}: {levels}")

    # drop the scalar level coordinate, otherwise the two terms disagree on it
    theta_upper = potential_temperature(ta.sel({plev_dim: upper}), upper_pa, p0_pa) \
        .drop_vars(plev_dim, errors='ignore')
    theta_surface = potential_temperature(tas, ps, p0_pa)
    return (theta_upper - theta_surface).rename('LTS')


def to_decades(obj, years: int = 10):
    """Average consecutive blocks of ``years`` along time, dropping any remainder."""
    return obj.coarsen(time=years, boundary="trim").mean()


def decades_to_members(field_dec, pdv_dec, amv_dec, name: str = 'SST') -> xr.Dataset:
    """Turn decadal piControl fields into a classification-ready dataset.

    Each decade becomes one ``member``, so that everything downstream (the
    classification, the t-test, the composites) is identical to the LE case
    where ``member`` really is an ensemble member.

    Only the classification notebook calls this.  Every other notebook consumes
    an already saved index through ``attach_index``.
    """
    ds = field_dec.to_dataset(name=name).rename({'time': 'member'})
    # Keep the decade midpoint year.  It is the only way a notebook that later
    # reads the saved index can verify it is looking at the same decades.
    ds = ds.assign_coords(
        member=np.arange(ds.sizes['member']),
        year=('member', field_dec['time'].dt.year.values),
    )
    ds['PDV'] = ('member', pdv_dec.values)
    ds['AMV'] = ('member', amv_dec.values)
    return ds


def attach_index(field_dec, index: xr.Dataset, name: str) -> xr.Dataset:
    """Put a decadal field onto the sample axis of an existing storyline index.

    The counterpart to ``decades_to_members`` for every notebook that *uses* the
    classification instead of producing it.  PDV, AMV and IVS come from the
    index file, never from ``field_dec`` -- the storylines are a property of the
    decades, and recomputing them from e.g. LTS would be meaningless.

    Raises if the decades do not line up, because otherwise the labels would be
    attached to the wrong decades with no visible error.
    """
    years = field_dec['time'].dt.year.values
    expected = index['year'].values
    if not np.array_equal(years, expected):
        raise ValueError(
            f"decade mismatch: field has {len(years)} decades "
            f"({years[0]}-{years[-1]}), index has {len(expected)} "
            f"({expected[0]}-{expected[-1]})"
        )

    ds = field_dec.to_dataset(name=name).rename({'time': 'member'})
    ds = ds.assign_coords(member=index['member'].values, year=('member', years))
    for var in ('PDV', 'AMV', 'IVS'):
        ds[var] = index[var]
    return ds


def prepare_variable(da, index: xr.Dataset, name: str,
                     mask: Optional[xr.DataArray] = None,
                     freq: str = 'YS', years: int = 10,
                     load: bool = True) -> xr.Dataset:
    """Raw monthly piControl field -> decadal anomalies carrying the storylines.

    The complete per-variable pipeline in one call: annual means, optional mask,
    spin-up dropped and drift removed, averaged into decades, index attached.
    Pass ``mask=None`` for atmospheric fields, which are defined over land too.

    ``load=True`` evaluates the result once, here.  
    """
    anom = annual_anomalies_picontrol(da, mask=mask, freq=freq)
    out = attach_index(to_decades(anom, years=years), index, name=name)
    return out.compute() if load else out


def attach_index_le(field, index: xr.Dataset, name: str) -> xr.Dataset:
    """Put an LE field onto the sample axis of an existing storyline index.

    The LE counterpart of ``attach_index``.  There the join key is the decade
    midpoint year; here the sample dimension already *is* the member label that
    ``load_le_members`` reads out of the filename, so the join is on ``member``.

    PDV, AMV and IVS always come from the index file, never from ``field`` --
    the storylines are a property of the members, and recomputing them from e.g.
    LTS would be meaningless.

    The index is reordered onto the members of ``field``, so a different file
    ordering is harmless; a genuine mismatch raises instead of silently
    labelling the wrong members.
    """
    if isinstance(field, xr.DataArray):
        ds = field.rename(name).to_dataset()
    else:
        ds = field.copy()

    members = np.asarray(ds['member'].values)
    expected = np.asarray(index['member'].values)

    if set(members.tolist()) != set(expected.tolist()):
        raise ValueError(
            f"member mismatch: field has {len(members)} members "
            f"({members[0]}..{members[-1]}), index has {len(expected)} "
            f"({expected[0]}..{expected[-1]})"
        )

    index = index.sel(member=xr.DataArray(members, dims='member'))
    for var in ('PDV', 'AMV', 'IVS'):
        ds[var] = index[var]
    return ds


def prepare_variable_le(da, index: xr.Dataset, name: str,
                        mask: Optional[xr.DataArray] = None,
                        load: bool = True) -> xr.Dataset:
    """Raw monthly LE field -> decadal anomalies carrying the storylines.

    The LE counterpart of ``prepare_variable``: annual means, optional mask,
    anomalies about the ensemble mean (= forced response removed), then the mean
    over the fixed decade, then the saved index attached.  Pass ``mask=None``
    for atmospheric fields, which are defined over land too.

    ``load=True`` evaluates the result once, here -- same reasoning as in
    ``prepare_variable``: the decadal field is a few MB per member while the
    graph behind it can be tens of GB.
    """
    anom = annual_anomalies_le(da, mask=mask)
    out = attach_index_le(anom.mean('time'), index, name=name)
    return out.compute() if load else out


# ---------------------------------------------------------------------------
# variability indices
# ---------------------------------------------------------------------------

def compute_pdv_index(tos: xr.DataArray) -> xr.DataArray:
    """PDV index from the Pacific tripole (equatorial minus north and south lobes).

    Expects SST anomalies on the native 0..360 longitude grid.
    """
    w = area_weights(tos["lat"])

    equatorial_box = tos.where(
        (tos.lat >= -10) & (tos.lat <= 10) & (tos.lon >= 170) & (tos.lon <= 270),
        drop=True,
    )
    north_pacific_box = tos.where(
        (tos.lat >= 25) & (tos.lat <= 45) & (tos.lon >= 140) & (tos.lon <= 215),
        drop=True,
    )
    south_pacific_box = tos.where(
        (tos.lat >= -50) & (tos.lat <= -15) & (tos.lon >= 150) & (tos.lon <= 200),
        drop=True,
    )

    tos_equatorial = equatorial_box.weighted(w).mean(("lat", "lon"))
    tos_north = north_pacific_box.weighted(w).mean(("lat", "lon"))
    tos_south = south_pacific_box.weighted(w).mean(("lat", "lon"))

    return (tos_equatorial - 0.5 * (tos_north + tos_south)).rename("PDV")


def compute_amv_index(tos: xr.DataArray) -> xr.DataArray:
    """AMV index: North Atlantic minus near-global SST anomaly.

    North Atlantic 0-60N, 280-360E; global reference 60S-60N.
    Expects SST anomalies on the native 0..360 longitude grid.
    """
    w = area_weights(tos["lat"])

    natl = tos.sel(lat=slice(0, 60), lon=slice(280, 360))
    natl_mean = natl.weighted(w).mean(("lat", "lon"))

    glob = tos.sel(lat=slice(-60, 60), lon=slice(0, 360))
    glob_mean = glob.weighted(w).mean(("lat", "lon"))

    return (natl_mean - glob_mean).rename("AMV")


# ---------------------------------------------------------------------------
# storyline classification
# ---------------------------------------------------------------------------

def classify_and_plot(amv_list, pdv_list, alpha=20,
                      ellipse_levels=(0.2,), #0.5, 0.8, 0.95
                      classify='transformed',
                      save_path=None):
    """Classify the 2-D PDV/AMV distribution and draw the PCA-based scatter plot.

    ``alpha`` is the size of the neutral class in percent, applied as a
    Mahalanobis confidence ellipse: points inside the alpha% ellipse are
    neutral (class 0), points outside fall into one of the four storylines.

    ``classify='transformed'`` (default) puts the dividing lines along the axes
    of maximal variance, which accounts for the PDV/AMV correlation and gives
    roughly equal class sizes.  ``classify='quadrants'`` ignores that
    correlation but isolates the individual effect of each driver better.

    Returns ``(normalized_data, classifications, x, y, IVS_labels, IVS_colors)``
    where ``x`` is standardized PDV and ``y`` is standardized AMV.
    """
    print(f"Classifying data with alpha={alpha}...")

    # Step 1: flatten and standardize -- x is PDV, y is AMV
    pdv = np.concatenate(pdv_list)
    amv = np.concatenate(amv_list)
    x = (pdv - pdv.mean()) / pdv.std(ddof=1) #
    y = (amv - amv.mean()) / amv.std(ddof=1) #
    normalized_data = np.vstack((x, y)).T

    r = np.corrcoef(x, y)[0, 1]

    # Step 2: PCA on the standardized cloud
    cov_matrix = np.cov(normalized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    transform_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # reference vectors in PCA space -> the two dividing lines
    v1_transformed = transform_matrix @ np.array([1, 1])
    v2_transformed = transform_matrix @ np.array([1, -1])

    # Step 3: neutral class = inside the alpha% confidence ellipse
    whitened = (normalized_data @ eigenvectors) / np.sqrt(eigenvalues)
    distances = np.linalg.norm(whitened, axis=1)      # Mahalanobis radius
    cutoff = np.sqrt(chi2.ppf(alpha / 100, df=2))
    keep_mask = distances > cutoff

    # Step 4: vectorized classification
    def enforce_direction(vec, ref_axis):
        return vec if vec[ref_axis] >= 0 else -vec

    v1_norm = enforce_direction(v1_transformed / np.linalg.norm(v1_transformed), ref_axis=0)
    v2_norm = enforce_direction(v2_transformed / np.linalg.norm(v2_transformed), ref_axis=1)

    v1_normal = np.array([-v1_norm[1], v1_norm[0]])
    v2_normal = np.array([-v2_norm[1], v2_norm[0]])

    classifications = np.zeros(len(normalized_data), dtype=int)  # neutral = 0

    if classify == 'quadrants':
        points_to_class = normalized_data[keep_mask]
        angles = np.arctan2(points_to_class[:, 1], points_to_class[:, 0])
        quad_labels = np.zeros(len(angles), dtype=int)
        quad_labels[(angles >= 0) & (angles < np.pi / 2)] = 1        # top-right
        quad_labels[(angles >= np.pi / 2) & (angles <= np.pi)] = 2   # top-left
        quad_labels[(angles < -np.pi / 2)] = 3                       # bottom-left
        quad_labels[(angles >= -np.pi / 2) & (angles < 0)] = 4       # bottom-right
        classifications[keep_mask] = quad_labels

    elif classify == 'transformed':
        # signed distance to each dividing line
        p1 = (normalized_data @ v1_normal)[keep_mask]
        p2 = (normalized_data @ v2_normal)[keep_mask]
        s1, s2 = p1 >= 0, p2 >= 0

        # calibrate the sign pairs against the four reference quadrants so the
        # labels stay tied to the physical phases whatever the rotation is
        ref_points = np.array([
            [1.0, 1.0],    # PDV+/AMV+ -> 1
            [-1.0, 1.0],   # PDV-/AMV+ -> 2
            [-1.0, -1.0],  # PDV-/AMV- -> 3
            [1.0, -1.0],   # PDV+/AMV- -> 4
        ])
        ref_signs = list(zip(ref_points @ v1_normal >= 0, ref_points @ v2_normal >= 0))
        signpair_to_label = dict(zip(ref_signs, [1, 2, 3, 4]))

        classifications[keep_mask] = np.array(
            [signpair_to_label[(a, b)] for a, b in zip(s1, s2)], dtype=int
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, color in enumerate(IVS_colors):
        mask_class = classifications == i
        ax.scatter(normalized_data[mask_class, 0], normalized_data[mask_class, 1],
                   c=color, s=30, alpha=0.5, label=IVS_labels[i])

    # dividing lines
    max_len = 3
    for v in [v1_norm, v2_norm]:
        ax.plot([-v[0] * max_len, v[0] * max_len],
                [-v[1] * max_len, v[1] * max_len],
                color='black', lw=1)

    # confidence ellipses
    ellipse_styles = ['-', '--', ':', (0, (3, 5, 1, 5))]
    ellipse_handles = []
    for i, level in enumerate(ellipse_levels):
        radius = np.sqrt(chi2.ppf(level, df=2))
        style = ellipse_styles[i % len(ellipse_styles)]
        ax.add_patch(patches.Ellipse(
            (0, 0),
            width=2 * radius * np.sqrt(eigenvalues[0]),
            height=2 * radius * np.sqrt(eigenvalues[1]),
            angle=np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
            edgecolor='grey', facecolor='none', lw=1.2, linestyle=style,
        ))
        ellipse_handles.append(patches.Patch(
            edgecolor='grey', facecolor='none', lw=1.2, linestyle=style,
            label=f"{int(level * 100)}% Confidence Ellipse",
        ))

   # ellipse_legend = ax.legend(handles=ellipse_handles, loc='lower right',
   #                            title="Confidence Levels")
   # ax.add_artist(ellipse_legend)

    # class means
    for i in range(1, 5):
        mask_class = classifications == i
        if np.any(mask_class):
            mean_x, mean_y = normalized_data[mask_class].mean(axis=0)
            ax.scatter(mean_x, mean_y, marker='x', color='black', s=80, lw=2, zorder=3)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("PDV [σ]")
    ax.set_ylabel("AMV [σ]")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved classification plot to: {save_path}")
    else:
        plt.show()
    plt.close()

    print(f"PDV-AMV correlation: r = {r:+.3f}")
    for i in range(5):
        print(f"{IVS_labels[i]}: {np.sum(classifications == i)} points")

    return normalized_data, classifications, x, y, IVS_labels, IVS_colors


def ivs_geometry(pdv, amv, alpha=20, classify='transformed') -> dict:
    """The classification of ``classify_and_plot`` without the figure.

    Same steps 1-4 (standardise with ddof=1, PCA, neutral class inside the
    alpha % Mahalanobis ellipse, dividing lines along the axes of maximal
    variance), so the classes are identical. Returns what a figure of the
    classification needs: ``x``, ``y`` (standardised PDV and AMV), ``classes``,
    ``eigenvalues``, ``eigenvectors`` and the two dividing directions ``lines``.
    """
    pdv, amv = np.asarray(pdv, dtype=float), np.asarray(amv, dtype=float)
    x = (pdv - pdv.mean()) / pdv.std(ddof=1)
    y = (amv - amv.mean()) / amv.std(ddof=1)
    data = np.vstack((x, y)).T

    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(data, rowvar=False))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    transform_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    v1 = transform_matrix @ np.array([1, 1])
    v2 = transform_matrix @ np.array([1, -1])

    whitened = (data @ eigenvectors) / np.sqrt(eigenvalues)
    keep = np.linalg.norm(whitened, axis=1) > np.sqrt(chi2.ppf(alpha / 100, df=2))

    def direction(vec, ref_axis):
        return vec if vec[ref_axis] >= 0 else -vec

    v1_norm = direction(v1 / np.linalg.norm(v1), 0)
    v2_norm = direction(v2 / np.linalg.norm(v2), 1)
    v1_normal = np.array([-v1_norm[1], v1_norm[0]])
    v2_normal = np.array([-v2_norm[1], v2_norm[0]])

    classes = np.zeros(len(data), dtype=int)
    if classify == 'quadrants':
        angles = np.arctan2(data[keep, 1], data[keep, 0])
        labels = np.zeros(len(angles), dtype=int)
        labels[(angles >= 0) & (angles < np.pi / 2)] = 1
        labels[(angles >= np.pi / 2) & (angles <= np.pi)] = 2
        labels[angles < -np.pi / 2] = 3
        labels[(angles >= -np.pi / 2) & (angles < 0)] = 4
        classes[keep] = labels
    else:
        s1, s2 = (data @ v1_normal)[keep] >= 0, (data @ v2_normal)[keep] >= 0
        ref = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        ref_signs = list(zip(ref @ v1_normal >= 0, ref @ v2_normal >= 0))
        lookup = dict(zip(ref_signs, [1, 2, 3, 4]))
        classes[keep] = np.array([lookup[(a, b)] for a, b in zip(s1, s2)], dtype=int)

    return {'x': x, 'y': y, 'classes': classes, 'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors, 'lines': (v1_norm, v2_norm), 'alpha': alpha}


def add_ivs(ds: xr.Dataset, classifications, dim: str = 'member') -> xr.Dataset:
    """Attach the storyline classes to a dataset as the ``IVS`` variable."""
    ds['IVS'] = xr.DataArray(
        np.asarray(classifications),
        dims=[dim],
        coords={dim: ds[dim]},
        attrs={'description': 'IVS classification labels'},
    )
    return ds


def storyline_labels(ivs) -> np.ndarray:
    """Map integer classes to their label strings.

    Handy for building the dataframes the boxplot and scatter sections use::

        df['storyline'] = storyline_labels(decadal_means['IVS'].values)
    """
    values = ivs.values if isinstance(ivs, xr.DataArray) else np.asarray(ivs)
    return np.asarray(IVS_labels)[values.astype(int)]


# ---------------------------------------------------------------------------
# saving and reloading, so that nothing is classified twice
# ---------------------------------------------------------------------------

def save_ivs_index(ds: xr.Dataset, path, **settings) -> xr.Dataset:
    """Write the PDV/AMV/IVS table -- a few kB that every other notebook reads.

    Only the classification notebook should call this.  ``settings`` records
    what the labels depend on (alpha, classify, clip_years, decade_years) so a
    later mismatch is visible instead of silent.

    What identifies a sample differs between the two setups, so only piControl
    is required to carry a ``year`` coordinate:

        piControl : sample = decade, identified by its midpoint ``year``, which
                    ``decades_to_members`` attaches
        LE        : sample = ensemble member, identified by the ``member`` label
                    itself, so there is nothing extra to check
    """
    index = ds[['PDV', 'AMV', 'IVS']].copy()
    index['IVS'] = index['IVS'].astype('int8')
    index.attrs = {**ds.attrs,
                   **{k: v for k, v in settings.items() if v is not None}}

    if 'year' not in index.coords and np.issubdtype(index['member'].dtype, np.number):
        raise ValueError(
            "neither a 'year' coordinate nor member labels - the saved index "
            "could not be checked against a field later. Rebuild ds with "
            "decades_to_members (piControl) or load_le_members (LE)."
        )

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_netcdf(path)

    kind = 'decades' if 'year' in index.coords else 'members'
    print(f"wrote {path}  ({index.sizes['member']} {kind})")
    return index


def load_ivs_index(path=None, name: str = 'SST',
                   experiment: str = 'piControl') -> xr.Dataset:
    """Read a storyline index fully into memory and report how it was built.

    Pass ``experiment='LE'`` for the ensemble-member index, which has member
    labels instead of decade years.
    """
    path = pathlib.Path(path) if path is not None else index_path(name, experiment)
    with xr.open_dataset(path) as ds:
        index = ds.load()

    if 'year' in index.coords:
        span = f"{index['year'].values[0]}-{index['year'].values[-1]}"
        kind = 'decades'
    else:
        members = index['member'].values
        span = f"{members[0]}..{members[-1]}"
        kind = 'members'

    settings = ', '.join(f"{k}={v}" for k, v in index.attrs.items())
    print(f"{path.name}: {index.sizes['member']} {kind}, {span}\n  {settings}")
    return index


def save_decadal(ds: xr.Dataset, path, dtype: str = 'float32'):
    """Write a prepared decadal dataset for the cross-variable notebooks.

    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {'dtype': dtype, 'zlib': True, 'complevel': 4}
                for v in ds.data_vars if ds[v].ndim > 1}
    ds.to_netcdf(path, encoding=encoding)
    print(f"wrote {path}")
    return path


def load_decadal(*variables: str, experiment: str = 'piControl') -> xr.Dataset:
    """Merge prepared decadal fields onto their shared sample axis.

    ``load_decadal('SST', 'LTS', 'LCC')`` gives one dataset holding all three
    fields plus PDV, AMV and IVS.
    """
    return xr.merge([xr.open_dataset(decadal_path(v, experiment))
                     for v in variables])


def available_decadal(experiment: str = 'piControl') -> list:
    """Names of the prepared decadal fields, i.e. what ``load_decadal`` can take.

    Each name corresponds to one file written by a per-variable notebook.  The
    cross-variable notebooks call this first so that a missing field shows up as
    a readable message instead of a FileNotFoundError halfway down.
    """
    folder = data_root / experiment / 'decadal'
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob('*.nc'))


# ---------------------------------------------------------------------------
# significance testing
# ---------------------------------------------------------------------------

def fdr_sig_masks(
    field: xr.DataArray,
    ivs: xr.DataArray,
    alpha: float = 0.10,
    dim: str = 'member',
    method: str = 'indep',
    return_pvals: bool = False,
):
    """Per-storyline one-sample t-test against zero, with FDR correction.

    For each class 1-4 the composite is tested grid point by grid point across
    the sample dimension, and the resulting p-value field is FDR-corrected over
    all valid (ocean) points.

    Note on ``alpha``: under FDR this is the expected false-discovery rate
    *among the flagged points*, not a per-grid-point error rate.  Wilks (2016)
    recommends alpha_FDR = 2 * alpha_global for field significance, hence the
    default of 0.10 for a nominal 5% level.

    Returns ``sig_masks`` (dict class -> boolean DataArray), or
    ``(sig_masks, p_adj_maps)`` if ``return_pvals=True``.
    """
    sig_masks, p_adj_maps = {}, {}

    for i in range(1, 5):
        da = field.isel(**{dim: (ivs == i)})

        _, p_vals = stats.ttest_1samp(da.values, popmean=0, axis=0, nan_policy='omit')

        # flatten and keep only valid (ocean) grid points for the correction
        p_flat = p_vals.ravel()
        valid = np.isfinite(p_flat)

        reject = np.zeros(p_flat.shape, dtype=bool)
        p_adj = np.full(p_flat.shape, np.nan)
        rej, p_corr = fdrcorrection(p_flat[valid], alpha=alpha, method=method)
        reject[valid] = rej
        p_adj[valid] = p_corr

        coords = {'lat': da.lat, 'lon': da.lon}
        sig_masks[i] = xr.DataArray(reject.reshape(p_vals.shape),
                                    dims=['lat', 'lon'], coords=coords)
        p_adj_maps[i] = xr.DataArray(p_adj.reshape(p_vals.shape),
                                     dims=['lat', 'lon'], coords=coords)

    return (sig_masks, p_adj_maps) if return_pvals else sig_masks


# ---------------------------------------------------------------------------
# composites and plotting
# ---------------------------------------------------------------------------

def composite_means(ds, ivs=None, dim: str = 'member',
                    classes: Sequence[int] = (1, 2, 3, 4)) -> Dict[int, xr.Dataset]:
    """Mean field per storyline, ready to hand to ``plot_ivs_2x2``."""
    if ivs is None:
        ivs = ds['IVS']
    return {i: ds.where(ivs == i, drop=True).mean(dim) for i in classes}


def _crop_lat(ax, lat_limit: Optional[float]) -> None:
    """Show all longitudes but only ``-lat_limit..lat_limit``; ``None`` = whole globe.

    Only the view is cropped, the data is untouched.  The map stays global in
    x and only its y-limits are pulled in, in the projection's own coordinates:
    ``set_extent`` with a full -180..180 box loses part of the longitudes on a
    map centred on 180, because both box edges fall on the same meridian.
    """
    ax.set_global()
    if lat_limit is None:
        return
    centre = ax.projection.proj4_params.get('lon_0', 0.0)
    _, y = ax.projection.transform_point(centre, lat_limit, ccrs.PlateCarree())
    ax.set_ylim(-y, y)


def global_mean_labels(mean_fields: Dict[int, xr.Dataset], variable: str,
                       units: str = '') -> Dict[int, str]:
    """Area-weighted global mean of ``variable`` per storyline, as panel labels.

    Meant for ``plot_ivs_2x2(..., annotate=...)``.  The maps cannot show this
    number: large regional anomalies cancel, so for TOA it is what says whether
    a storyline leaves the planet gaining or losing energy.

    ``mean_fields`` does not have to be the composite being plotted -- pass the
    TOA composites to label the SST maps with the global-mean TOA of the same
    storylines.  The mean is always over the whole globe, whatever ``lat_limit``
    the map is cropped to.
    """
    labels = {}
    for ivs_class, field in mean_fields.items():
        w = area_weights(field['lat'])
        value = float(field[variable].weighted(w).mean(('lat', 'lon')))
        labels[ivs_class] = f"global mean {variable} {value:+.3f} {units}".rstrip()
    return labels


def plot_ivs_2x2(
    mean_fields: Dict[int, xr.DataArray],
    variable: str = 'SST',
    sig_masks: Dict[int, xr.DataArray] = None,
    save_path=None,
    cmap='RdBu_r',
    lat_limit: Optional[float] = 60,
    annotate: Optional[Dict[int, str]] = None,
    **kwargs
) -> plt.Figure:
    """2x2 composite map of the four storylines, for any variable.

    ``mean_fields`` maps storyline class integers to datasets containing
    ``variable`` (see ``composite_means``).  ``sig_masks`` optionally maps the
    same integers to boolean (lat, lon) arrays; True is stippled.

    ``cmap`` stays on ``RdBu_r``, which suits the fields in K.  Set it per
    variable in the notebook, next to ``levels``, the same way -- anything
    matplotlib accepts works, including a Colormap object.  It has to be a
    *diverging* map with a neutral centre, because every field here is an
    anomaly; a sequential one puts all the visual weight on one sign.

    ``save_path`` is off by default -- nothing is written unless one is given,
    e.g. ``save_path=figure_path('LTS_composites')``.  The figure is returned
    and still displays inline either way; it is deliberately not closed.

    ``lat_limit`` crops the maps to that latitude on both sides.  Nothing here
    is analysed poleward of 60 degrees, and sea ice makes the anomalies there
    large enough to pull the eye away from the tropics.  ``None`` gives the
    whole globe back.

    ``annotate`` maps storyline class integers to a line of text written under
    the bottom-left corner of that panel, e.g. from ``global_mean_labels``.

    Extra keyword arguments: ``levels``, ``figsize``, ``projection``,
    ``cbar_label``, ``add_titles``, ``panel_labels`` (default True: (a)-(d)
    above the maps, not bold), and for the stippling ``stipple_step``
    (a dot on every n-th grid point in lat and lon, default 4),
    ``stipple_size`` (default 1.5) and ``stipple_color`` (default '0.25').
    """
    expected_ivs_classes = [1, 2, 3, 4]

    fig, axes = plt.subplots(
        2, 2,
        figsize=kwargs.pop('figsize', (10, 8)),
        subplot_kw={"projection": kwargs.get(
    #        'projection', ccrs.Robinson(central_longitude=180.))},
            'projection', ccrs.EqualEarth(central_longitude=180.))},

    )
    axes = axes.flatten()
    # top row shows the two AMV+ phases, bottom row the two AMV- phases
    axes_reordered = [axes[1], axes[0], axes[2], axes[3]]

    im = None
    for ivs_class, ax in zip(expected_ivs_classes, axes_reordered):
        if ivs_class not in mean_fields:
            ax.text(0.5, 0.5, f"No data\nfor IVS {ivs_class}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
            _crop_lat(ax, lat_limit)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="lightgray")
            if kwargs.get('add_titles', True):
                ax.set_title(IVS_labels[ivs_class], fontsize=12, color="gray")
            continue

        field = mean_fields[ivs_class][variable]

        lon, lat = np.meshgrid(field.lon, field.lat)
        im = ax.contourf(
            lon, lat, field,
            levels=kwargs.get('levels', np.linspace(-.5, .5, 21)),
            transform=ccrs.PlateCarree(),
            transform_first=True,
            cmap=cmap,
            extend="both",
        )

        # overlay significance if provided: a small grey dot on every
        # stipple_step-th significant grid point, so the colours stay readable
        if sig_masks is not None and ivs_class in sig_masks:
            step = kwargs.get('stipple_step', 4)
            thinned = (slice(step // 2, None, step), slice(step // 2, None, step))
            mask = np.asarray(sig_masks[ivs_class]).astype(bool)[thinned]
            ax.scatter(
                lon[thinned][mask], lat[thinned][mask],
                s=kwargs.get('stipple_size', 1.5),
                color=kwargs.get('stipple_color', '0.25'),
                linewidths=0,
                transform=ccrs.PlateCarree(),
            )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        _crop_lat(ax, lat_limit)

        for spine in ax.spines.values():
            spine.set_edgecolor(IVS_colors[ivs_class])
            spine.set_linewidth(3)

        if kwargs.get('add_titles', True):
            ax.set_title(IVS_labels[ivs_class], fontsize=12, fontweight="bold")

        # under the panel rather than inside it: cropped to lat_limit the map
        # fills its axes, so a label in the corner would sit on top of data
        if annotate is not None and ivs_class in annotate:
            ax.text(0.01, -0.04, annotate[ivs_class], transform=ax.transAxes,
                    fontsize=9, va='top', ha='left')

    if im is None:
        raise ValueError("No plottable data found in mean_fields for classes 1-4.")

    # (a)-(d) top left above each map, in reading order; the colorbar gets none
    if kwargs.get('panel_labels', True):
        for ax, letter in zip(axes, 'abcd'):
            ax.text(0, 1.02, f'({letter})', transform=ax.transAxes,
                    fontsize=11, ha='left', va='bottom')

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.14, shrink=0.6)
    cbar.set_label(kwargs.get('cbar_label', f"{variable} anomaly"), fontsize=12)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"wrote {save_path}")

    return fig


# ---------------------------------------------------------------------------
# region analysis: relationships between variables
# ---------------------------------------------------------------------------

def region_frame(ds: xr.Dataset, variables: Sequence[str],
                 boxes: Optional[Sequence[str]] = None,
                 coarsen: Optional[int] = 2,
                 mask: Optional[xr.DataArray] = None,
                 min_ocean: float = 0.0) -> pd.DataFrame:
    """Tidy frame for the region plots: one row per grid point, sample and box.

    Columns: region, member, storyline, lat, lon, weight and one per variable.
    ``member`` is kept so that both views come out of the same frame -- every
    grid point, or the box mean per sample via ``region_means``.

    ``weight`` is the area weight of the cell from ``box_weights`` and is what
    ``region_means`` averages with; it is carried in the frame rather than
    recomputed so that every downstream figure and table uses the same numbers.

    ``mask`` (True = ocean, from ``ocean_mask``) is worth passing whenever the
    boxes contain land. The prepared decadal files keep the full fields -- only
    SST is masked at source, the atmospheric ones are not -- so without a mask
    here the two ends of a link are averaged over different areas: ocean-only
    SST against ocean-and-land LTS. Masking at this stage is a switch, not a
    recompute.

    The argument is called ``boxes`` rather than ``regions`` so that it does not
    shadow the module-level ``regions`` dict it defaults to.
    """
    if boxes is None:
        boxes = list(regions)
    variables = list(variables)

    # A mask on a different grid would align to nothing and silently empty the
    # merge below, so it is checked and then rebuilt on the data's own
    # coordinates -- floats that merely round the same are not good enough.
    if mask is not None:
        mask = mask.transpose('lat', 'lon')
        if not (np.array_equal(mask['lat'].values, ds['lat'].values)
                and np.array_equal(mask['lon'].values, ds['lon'].values)):
            raise ValueError(
                f"mask grid {mask.sizes} does not match the data grid "
                f"{{'lat': {ds.sizes['lat']}, 'lon': {ds.sizes['lon']}}} - "
                f"use the land-fraction file belonging to this experiment"
            )
        mask = xr.DataArray(mask.values, dims=['lat', 'lon'],
                            coords={'lat': ds['lat'], 'lon': ds['lon']})

    # The index is per sample, so it is joined on member instead of being
    # coarsened along with the fields.
    index = (ds[['PDV', 'AMV', 'IVS']].to_dataframe().reset_index()
             [['member', 'PDV', 'AMV', 'IVS']])

    frames = []
    for name in boxes:
        box = select_box(ds[variables], region=name, coarsen=coarsen, mask=mask)
        weight = box_weights(ds, region=name, coarsen=coarsen, mask=mask,
                             min_ocean=min_ocean)

        frame = box.to_dataframe().reset_index()
        frame = frame.merge(weight.to_dataframe().reset_index(),
                            on=['lat', 'lon'])
        frame['region'] = name
        frames.append(frame.merge(index, on='member'))

    out = pd.concat(frames, ignore_index=True)
    out['storyline'] = storyline_labels(out['IVS'].values)
    return out


def _draw_fit(ax, xv, yv, ci: Optional[float] = 0.95, band: bool = True):
    """Least-squares line, with the confidence band of the line if asked.

    The band is the interval of the **fitted line**, narrowest in the middle of
    the x range and flaring outwards because the slope levers hardest there.
    Read it by laying a horizontal line against it: one that fits inside over
    the whole width means the slope is not distinguishable from zero.  That is
    the same statement as p > 0.05, give or take -- a pointwise band is not
    exactly a test of the whole line, so p is the strict version and the band is
    the one you can see.  It earns its ink on the weak links and says nothing
    new on the tight ones.

    It is *not* a prediction interval, which would be "where does the next
    decade land" and about as wide as the point cloud itself.

    Returns ``(slope, r, p, n)``, or ``None`` if there is nothing to fit.
    """
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    n = xv.size
    if n < 3 or xv.min() == xv.max():
        return None

    slope, intercept, r, p, _ = stats.linregress(xv, yv)
    xx = np.linspace(xv.min(), xv.max(), 200)
    fitted = slope * xx + intercept
    ax.plot(xx, fitted, color='0.25', lw=1.8, zorder=6)

    if band and ci:
        resid = yv - (slope * xv + intercept)
        scatter_sd = np.sqrt((resid ** 2).sum() / (n - 2))
        sxx = ((xv - xv.mean()) ** 2).sum()
        se = scatter_sd * np.sqrt(1 / n + (xx - xv.mean()) ** 2 / sxx)
        half = stats.t.ppf(0.5 + ci / 2, n - 2)
        ax.fill_between(xx, fitted - half * se, fitted + half * se,
                        color='0.5', alpha=0.25, lw=0, zorder=5)

    return slope, r, p, n


def _fit_label(r: float) -> str:
    """Corner annotation for a fit: r only.

    R-squared and p come from the same regression and, at fixed n, from r alone,
    so they add no information to the figure.  Slope, its confidence interval
    and p are in ``link_table``.
    """
    return f"r = {r:+.2f}"


def plot_region_scatter(df: pd.DataFrame, x: str, y: str,
                        boxes: Optional[Sequence[str]] = None,
                        by_decade: bool = False,
                        order: Optional[Sequence[str]] = None,
                        figsize: Optional[tuple] = None,
                        s: int = 12, alpha: float = 0.4,
                        fit: bool = True, means: bool = True,
                        band: bool = True, ci: float = 0.95,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None) -> plt.Figure:
    """Scatter two variables against each other, one panel per region.

    ``by_decade=True`` reduces each box to its mean per sample -- one point per
    decade (piControl) or member (LE), directly comparable to the storylines.
    The default draws every coarsened grid point instead, which shows the
    spatial spread inside the box but carries nothing like the sample size it
    appears to: grid points in one box are strongly correlated.  Which of the
    two is drawn, and how many points that is, goes into the panel title.

    ``fit`` adds one least-squares line over all points, ``band`` its 95%
    confidence interval (``ci``), and the corner carries r, R-squared and p --
    see ``_draw_fit`` and ``_fit_label`` for what each of those is for.

    """
    boxes = list(boxes) if boxes is not None else list(df['region'].unique())
    order = list(order) if order is not None else IVS_order
    colors = dict(zip(IVS_labels, IVS_colors))

    data = df
    if by_decade:
        data = region_means(df, [x, y], boxes=boxes)

    fig, axes = plt.subplots(1, len(boxes), sharex=True, sharey=True,
                             figsize=figsize or (5 * len(boxes), 4.5))
    axes = np.atleast_1d(axes)

    view = 'box means' if by_decade else 'grid points'

    for ax, name in zip(axes, boxes):
        sub = data[data['region'] == name]

        for label in order:
            pts = sub[sub['storyline'] == label]
            ax.scatter(pts[x], pts[y], s=s, alpha=alpha, color=colors[label],
                       label=label, edgecolors='none')

        # star at the centre of each storyline cloud -- where the storylines sit
        # relative to each other, without being hidden by the scatter
        if means:
            for label in order:
                pts = sub[sub['storyline'] == label]
                if len(pts):
                    ax.scatter(pts[x].mean(), pts[y].mean(), marker='*', s=260,
                               color=colors[label], edgecolors='black',
                               linewidths=0.8, zorder=7)

        # one fit, over all points
        if fit:
            fitted = _draw_fit(ax, sub[x].values, sub[y].values,
                               ci=ci, band=band)
            if fitted is not None:
                _, r, _, _ = fitted
                ax.text(0.03, 0.97, _fit_label(r), transform=ax.transAxes,
                        va='top', ha='left', fontsize=9, zorder=8,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.75, edgecolor='none'))

        ax.set_title(f"{name.replace('_', ' ')}  -  {view}, n={len(sub):,}",
                     fontsize=10)
        ax.set_xlabel(xlabel or x)

    axes[0].set_ylabel(ylabel or y)
    axes[-1].legend(fontsize=8, markerscale=2, frameon=False, loc='lower right')
    fig.tight_layout()
    return fig


def region_means(df: pd.DataFrame, variables: Sequence[str],
                 boxes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Box mean per sample: one row per region and decade (piControl) or member (LE).

    Every figure and table that shows a box mean goes through here, so that they
    cannot drift apart.

    The mean is **area weighted** whenever ``region_frame`` put a ``weight``
    column in the frame: cos(lat), times the ocean fraction of the cell if a
    mask was used.  Each variable gets its own denominator, built from the cells
    where *that* variable is finite -- SST is masked at source and the
    atmospheric fields are not, so they do not share a set of valid cells and a
    single shared denominator would quietly misweight them.

    A frame without a ``weight`` column falls back to the plain mean.
    """
    data = df if boxes is None else df[df['region'].isin(list(boxes))]
    variables = list(variables)
    keys = ['region', 'member', 'storyline']

    if 'weight' not in data.columns:
        return data.groupby(keys, as_index=False)[variables].mean()

    w = data['weight'].to_numpy(dtype=float)
    parts = data[keys].copy()
    for v in variables:
        x = data[v].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(w)
        parts[f'num_{v}'] = np.where(ok, x * w, 0.0)
        parts[f'den_{v}'] = np.where(ok, w, 0.0)

    agg = parts.groupby(keys, as_index=False).sum()

    out = agg[keys].copy()
    for v in variables:
        den = agg[f'den_{v}'].to_numpy()
        num = agg[f'num_{v}'].to_numpy()
        out[v] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def region_summary(df: pd.DataFrame, variables: Sequence[str],
                   boxes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The numbers behind ``plot_region_box``: one row per region, variable, storyline.

    Computed on the box means, so a row summarises samples -- decades in
    piControl, members in the LE -- and not grid points.

    ``median`` sits next to ``mean`` on purpose.  The boxplot draws the median
    and the interquartile range, this adds the mean and the standard deviation,
    and the two pairs only agree while the distribution is roughly symmetric.
    Where they part company the box is the honest picture and the std is a poor
    summary of it.

    ``std`` is the spread *within* one storyline -- the internal variability the
    storyline does not explain -- and is what the separation between storyline
    means has to be judged against.  ``sem = std / sqrt(n)`` is the uncertainty
    of the storyline mean itself, i.e. the error bar on the stars in
    ``plot_region_scatter``.
    """
    data = region_means(df, variables, boxes=boxes)

    frames = []
    for v in variables:
        out = (data.groupby(['region', 'storyline'])[v]
                   .agg(n='count', mean='mean', median='median', std='std')
                   .reset_index())
        out['sem'] = out['std'] / np.sqrt(out['n'])
        out.insert(0, 'variable', v)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def as_links(links: Sequence) -> list:
    """Normalise a chain or a list of pairs into a list of ``(x, y)`` links.

    ``['SST', 'LTS', 'CLDLOW']`` becomes ``[('SST', 'LTS'), ('LTS', 'CLDLOW')]``,
    while a list of pairs is passed through.  Pairs are what the radiative end of
    the chain needs: low cloud feeds SWTOA, LWTOA and the net TOA in parallel,
    which a single linear chain cannot express.
    """
    links = list(links)
    if links and isinstance(links[0], str):
        return list(zip(links[:-1], links[1:]))
    return [tuple(pair) for pair in links]


def link_variables(links: Sequence) -> list:
    """Every variable appearing in ``links``, in order of first appearance."""
    names = [name for pair in as_links(links) for name in pair]
    return list(dict.fromkeys(names))


def _fit(a: np.ndarray, b: np.ndarray,
         ci: float = 0.95) -> Tuple[float, float, float, float, float]:
    """Slope, intercept, correlation, CI half-width of the slope and p of b on a.

    Rows where either side is not finite are dropped from both, which is why the
    intercept has to come from here rather than from the two means separately --
    those are taken over different subsets whenever the NaN patterns differ.
    The CI and p treat every sample as independent.
    Returns all-NaN if fewer than three usable pairs are left.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return (np.nan,) * 5
    res = stats.linregress(a[ok], b[ok])
    half = stats.t.ppf(0.5 + ci / 2, n - 2) * res.stderr
    return res.slope, res.intercept, res.rvalue, half, res.pvalue


def link_table(df: pd.DataFrame, links: Sequence,
               boxes: Optional[Sequence[str]] = None,
               per_storyline: bool = True) -> pd.DataFrame:
    """Slope and correlation for every link, per region and storyline.

    ``links`` is either a chain (``['SST', 'LTS', 'CLDLOW']``) or an explicit
    list of pairs -- see ``as_links``.

    Always computed on the box means from ``region_means``, never on the raw grid
    points: those are spatially correlated, so an r over them describes the box
    pattern rather than carrying the sample size it appears to have.

    """
    links = as_links(links)
    dec = region_means(df, link_variables(links), boxes=boxes)
    boxes = list(boxes) if boxes is not None else list(dec['region'].unique())

    rows = []
    for name in boxes:
        sub = dec[dec['region'] == name]
        for x, y in links:
            groups = [('all', sub)]
            if per_storyline:
                groups += [(label, sub[sub['storyline'] == label])
                           for label in IVS_order]
            for label, g in groups:
                slope, _, r, half, p = _fit(g[x].values, g[y].values)
                rows.append({'region': name, 'link': f'{x} -> {y}',
                             'storyline': label, 'n': len(g),
                             'slope': slope, 'ci95': half, 'r': r, 'r2': r ** 2,
                             'p': p})

    return pd.DataFrame(rows)


def plot_region_map(boxes: Optional[Sequence[str]] = None,
                    figsize: tuple = (9, 4.5),
                    central_longitude: float = 180.0,
                    lat_limit: Optional[float] = 60,
                    save_path=None) -> plt.Figure:
    """Draw the analysis boxes on a world map -- the legend for every later figure.

    Boxes come from the module-level ``regions`` dict, so the map can never show
    a box that the analysis does not use.  Longitudes are stored in the 0..360
    convention of CESM2 and converted here, because cartopy's ``PlateCarree``
    wants -180..180.

    ``lat_limit`` crops the map like in ``plot_ivs_2x2``; ``None`` = whole globe.
    """
    boxes = list(boxes) if boxes is not None else list(regions)
    palette = ['tab:purple', 'tab:brown', 'tab:cyan', 'tab:olive']

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={
    #    'projection': ccrs.Robinson(central_longitude=central_longitude)})
        'projection': ccrs.EqualEarth(central_longitude=central_longitude)})
    _crop_lat(ax, lat_limit)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(alpha=0.3)

    for i, name in enumerate(boxes):
        lat0, lat1 = regions[name]['lat']
        lon0, lon1 = regions[name]['lon']
        x0 = lon0 - 360 if lon0 > 180 else lon0
        x1 = lon1 - 360 if lon1 > 180 else lon1
        color = palette[i % len(palette)]

        ax.add_patch(patches.Rectangle(
            (x0, lat0), x1 - x0, lat1 - lat0,
            transform=ccrs.PlateCarree(), zorder=5,
            facecolor=color, alpha=0.35, edgecolor=color, linewidth=2,
        ))
        ax.text(0.5 * (x0 + x1), lat1 + 2, name.replace('_', ' '),
                transform=ccrs.PlateCarree(), zorder=6,
                ha='center', va='bottom', fontsize=9,
                color=color, fontweight='bold')

    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"wrote {save_path}")

    return fig


def _oriented_boxplot(ax, groups, positions, horizontal: bool, **kwargs):
    """``ax.boxplot`` with an orientation, across matplotlib versions.

    3.10 renamed ``vert=`` to ``orientation=`` and deprecated the old spelling,
    so try the new one first and fall back rather than pinning a version.
    """
    try:
        return ax.boxplot(
            groups, positions=positions,
            orientation='horizontal' if horizontal else 'vertical', **kwargs)
    except TypeError:
        return ax.boxplot(groups, positions=positions,
                          vert=not horizontal, **kwargs)


def _joint_panel(sub, panel: pd.DataFrame, x: str, y: str,
                 order: Sequence[str],
                 s: int = 30, alpha: float = 0.8,
                 fit: bool = True, means: bool = True,
                 band: bool = True, ci: float = 0.95,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None):
    """One joint panel into ``sub``: scatter in the centre, margins top and right.

    ``panel`` holds the box means of one region.  Shared by
    ``plot_region_joint`` (one panel per region) and ``plot_region_chain`` (one
    panel per link), so both always look the same.  Returns the scatter axes,
    which is where the caller puts the legend.
    """
    order = list(order)
    colors = dict(zip(IVS_labels, IVS_colors))

    axes = sub.subplots(
        2, 2, sharex='col', sharey='row',
        gridspec_kw=dict(width_ratios=[3.4, 1.3], height_ratios=[1.3, 3.4],
                         wspace=0.06, hspace=0.06))
    ax_top, ax_dead = axes[0]
    ax_sc, ax_right = axes[1]
    ax_dead.set_visible(False)

    for label in order:
        pts = panel[panel['storyline'] == label]
        ax_sc.scatter(pts[x], pts[y], s=s, alpha=alpha, color=colors[label],
                      label=label, edgecolors='none')

    # anomalies, so the origin is a meaningful reference and worth marking
    ax_sc.axhline(0, color='grey', lw=0.7, ls='--', zorder=1)
    ax_sc.axvline(0, color='grey', lw=0.7, ls='--', zorder=1)

    # a star rather than a bigger dot: among this many small points, shape
    # is picked out at a glance where size has to be searched for
    if means:
        for label in order:
            pts = panel[panel['storyline'] == label]
            if len(pts):
                ax_sc.scatter(pts[x].mean(), pts[y].mean(), marker='*',
                              s=200, color=colors[label], edgecolors='black',
                              linewidths=0.8, zorder=7)

    if fit:
        fitted = _draw_fit(ax_sc, panel[x].values, panel[y].values,
                           ci=ci, band=band)
        if fitted is not None:
            _, r, _, _ = fitted
            ax_sc.text(0.03, 0.97, _fit_label(r),
                       transform=ax_sc.transAxes,
                       va='top', ha='left', fontsize=9, zorder=8,
                       bbox=dict(boxstyle='round', facecolor='white',
                                 alpha=0.75, edgecolor='none'))

    ax_sc.set_xlabel(xlabel or x)
    ax_sc.set_ylabel(ylabel or y)

    # the margins: same samples, one variable at a time. The boxplot runs
    # along the axis it shares with the scatter, so the boxes line up with
    # the cloud they summarise.
    positions = list(range(len(order)))
    for ax, variable, horizontal in ((ax_top, x, True), (ax_right, y, False)):
        groups = [panel.loc[panel['storyline'] == label, variable]
                       .dropna().values for label in order]
        # box = the middle 50%, whiskers 1.5 IQR, anything past them drawn as
        # a point.
        drawn = _oriented_boxplot(ax, groups, positions, horizontal,
                                  widths=0.75, patch_artist=True,
                                  medianprops={'color': 'black'})
        for patch, label in zip(drawn['boxes'], order):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.7)
        for flier, label in zip(drawn['fliers'], order):
            flier.set(marker='o', markersize=3, alpha=0.6,
                      markerfacecolor=colors[label], markeredgecolor='none')

        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # the storylines are identified by colour, so the categorical axis of
    # each margin carries no information and only costs space
    ax_top.set_yticks([])
    ax_right.set_xticks([])

    return ax_sc


def _storyline_legend(fig, ax):
    """One legend for the whole figure, centred underneath it.

    Outside the panels, so it never covers data, wherever the points happen to
    lie.  It sits just below the canvas; the inline backend and
    ``savefig(..., bbox_inches='tight')`` both grow the image to include it.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), frameon=False, fontsize=9, markerscale=2)


def plot_region_joint(df: pd.DataFrame, x: str, y: str,
                      boxes: Optional[Sequence[str]] = None,
                      order: Optional[Sequence[str]] = None,
                      figsize: Optional[tuple] = None,
                      s: int = 30, alpha: float = 0.8,
                      fit: bool = True, means: bool = True,
                      band: bool = True, ci: float = 0.95,
                      xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None) -> plt.Figure:
    """``plot_region_scatter`` with the two marginal boxplots attached.

    Per region: the box means of ``x`` against ``y`` in the centre, the ``x``
    distribution per storyline along the top, the ``y`` distribution per
    storyline down the right side.  All three are the same samples -- the
    scatter is the joint distribution and the two boxplots are its margins.

    Always the box-mean view; for grid points use ``plot_region_scatter`` with
    ``by_decade=False``.
    """
    boxes = list(boxes) if boxes is not None else list(df['region'].unique())
    order = list(order) if order is not None else IVS_order

    data = region_means(df, [x, y], boxes=boxes)

    fig = plt.figure(figsize=figsize or (5.5 * len(boxes), 5.4))
    subfigs = np.atleast_1d(fig.subfigures(1, len(boxes), wspace=0.10))

    ax_sc = None
    for sub, name in zip(subfigs, boxes):
        ax_sc = _joint_panel(sub, data[data['region'] == name], x, y, order,
                             s=s, alpha=alpha, fit=fit, means=means,
                             band=band, ci=ci, xlabel=xlabel, ylabel=ylabel)
        sub.suptitle(name.replace('_', ' '), fontsize=10)

    if ax_sc is not None:
        _storyline_legend(fig, ax_sc)
    return fig


def plot_region_chain(df: pd.DataFrame, region: str, links: Sequence,
                      label=None, ncols: Optional[int] = None,
                      order: Optional[Sequence[str]] = None,
                      figsize: Optional[tuple] = None,
                      title: Optional[str] = None,
                      s: int = 30, alpha: float = 0.8,
                      fit: bool = True, means: bool = True,
                      band: bool = True, ci: float = 0.95) -> plt.Figure:
    """A whole pathway for one region: one joint panel per link, in one figure.

    ``plot_region_joint`` turned around -- there one link across all regions,
    here one region across all links, so a pathway reads in one figure from SST
    to the top of the atmosphere.  The panels are drawn by the same
    ``_joint_panel``, so they look exactly like those in ``plot_region_joint``.

    ``links`` is a chain or a list of pairs, see ``as_links``.  ``label`` is a
    function from variable name to axis text (e.g. adding the unit); without it
    the plain name is used.  ``ncols`` defaults to at most three panels per row.
    ``title`` defaults to the region name; ``title=False`` leaves it out
    altogether, for figures that get a caption instead.
    """
    links = as_links(links)
    order = list(order) if order is not None else IVS_order
    text = label if label is not None else (lambda v: v)

    data = region_means(df, link_variables(links), boxes=[region])

    ncols = ncols or min(len(links), 3)
    nrows = int(np.ceil(len(links) / ncols))

    if title is False:
        fig = plt.figure(figsize=figsize or (5.5 * ncols, 5.6 * nrows))
        body = fig
    else:
        fig = plt.figure(figsize=figsize or (5.5 * ncols, 5.6 * nrows + 0.4))
        # a thin row on top for the title, so it does not sit on the panels
        head, body = fig.subfigures(2, 1, height_ratios=[0.4, 5.6 * nrows])
        head.text(0.5, 0.5,
                  title if title is not None else region.replace('_', ' '),
                  ha='center', va='center', fontsize=12, fontweight='bold')
    subfigs = np.atleast_1d(
        body.subfigures(nrows, ncols, wspace=0.10, hspace=0.08)).ravel()

    ax_sc = None
    for sub, (x, y) in zip(subfigs, links):
        ax_sc = _joint_panel(sub, data, x, y, order,
                             s=s, alpha=alpha, fit=fit, means=means,
                             band=band, ci=ci, xlabel=text(x), ylabel=text(y))

    if ax_sc is not None:
        _storyline_legend(fig, ax_sc)
    return fig


def plot_region_box(df: pd.DataFrame, variable: str,
                    boxes: Optional[Sequence[str]] = None,
                    by_decade: bool = True,
                    order: Optional[Sequence[str]] = None,
                    figsize: Optional[tuple] = None,
                    ylabel: Optional[str] = None) -> plt.Figure:
    """Distribution of one variable per storyline, one panel per region."""
    boxes = list(boxes) if boxes is not None else list(df['region'].unique())
    order = list(order) if order is not None else IVS_order
    colors = dict(zip(IVS_labels, IVS_colors))

    data = df
    if by_decade:
        data = region_means(df, [variable], boxes=boxes)

    fig, axes = plt.subplots(1, len(boxes), sharey=True,
                             figsize=figsize or (4 * len(boxes), 4.5))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, boxes):
        sub = data[data['region'] == name]
        groups = [sub.loc[sub['storyline'] == label, variable].dropna().values
                  for label in order]
        # box = the middle 50%, whiskers 1.5 IQR, the rest as points -- nothing
        # is hidden and the boxes keep a readable shape
        drawn = ax.boxplot(groups, patch_artist=True,
                           medianprops={'color': 'black'})
        for patch, label in zip(drawn['boxes'], order):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.7)
        for flier, label in zip(drawn['fliers'], order):
            flier.set(marker='o', markersize=3, alpha=0.6,
                      markerfacecolor=colors[label], markeredgecolor='none')
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
        ax.set_title(name.replace('_', ' '), fontsize=11)

    axes[0].set_ylabel(ylabel or variable)
    fig.tight_layout()
    return fig


# ===========================================================================
# Annual Lanczos-filtered analysis 
# ===========================================================================

from scipy.ndimage import convolve1d


# ---------------------------------------------------------------------------
# input files
# ---------------------------------------------------------------------------

def read_anom(name: str, anom_dir=data_root) -> xr.DataArray:
    """Annual anomalies of one variable, ``<anom_dir>/<name>_annual_anom.nc``.

    Written by 01_piControl_annual_anomalies.ipynb:
    annual means, first 100 years dropped, linear trend removed.
    """
    path = pathlib.Path(anom_dir) / f'{name}_annual_anom.nc'
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing: run 01_piControl_annual_anomalies.ipynb first')
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_dataarray(path, decode_times=time_coder).rename(name).load()


# ---------------------------------------------------------------------------
# filter and basic operations
# ---------------------------------------------------------------------------

def low_pass_weights(window, cutoff):
    """Lanczos low-pass weights (Duchon 1979)."""
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros(nwts)
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1.0, n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2.0 * np.pi * cutoff * k) / (np.pi * k)
    w[n - 1:0:-1] = firstfactor * sigma
    w[n + 1:-1] = firstfactor * sigma
    return w[1:-1]


def lanczos_lowpass(da, cutoff_years: float = 10, window: int = 41, dim: str = 'time'):
    """Lanczos low-pass along ``dim``, for a time series or a gridded field.

    The weights are symmetric, so nothing is shifted in time. The
    (window - 1) / 2 years at both ends that the filter cannot see completely
    are cut off.
    """
    weights = low_pass_weights(window, 1.0 / cutoff_years)
    half = len(weights) // 2
    values = convolve1d(da.values, weights, axis=da.get_axis_num(dim), mode='nearest')
    return da.copy(data=values).isel({dim: slice(half, -half)})


def global_mean(da):
    """Area-weighted (cos lat) mean over lat and lon."""
    return da.weighted(area_weights(da['lat'])).mean(('lat', 'lon'))


def standardise(da):
    return (da - da.mean()) / da.std()


def lag_label(lag):
    """Year relative to year 0: negative = TOA before, positive = TOA after."""
    return 'lag 0 yr' if lag == 0 else f'lag {lag:+d} yr'


def finish(fig, name, save: bool = False):
    """Show a figure; write it to figure_path(name) only if ``save`` is set."""
    if save:
        path = figure_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'wrote {path}')
    plt.show()


# ---------------------------------------------------------------------------
# effective sample size after Metz (1991), as in Miyamoto et al. (2026)
# ---------------------------------------------------------------------------

def acf(a, L):
    """Autocorrelation at the lags 1..L, normalised with the total variance.

    Time is the first axis, so this works for a series and for a field
    (time, lat, lon). The lag is the first axis of the result.
    """
    a = np.asarray(a)
    a = a - a.mean(axis=0)
    variance = (a * a).sum(axis=0)
    return np.stack([(a[tau:] * a[:-tau]).sum(axis=0) / variance for tau in range(1, L + 1)])


def decorrelation_time(acf_y, acf_x):
    """Effective decorrelation time T_e from two autocorrelation functions.

    T_e = 1 + 2 * sum_{tau=1..L} (1 - tau / L) * R_xx(tau) * R_yy(tau)
    (Metz 1991; eq. 2 of Miyamoto et al. 2026), limited to T_e >= 1 so that
    the effective sample size never exceeds the number of time steps.
    ``acf_x`` is a series, ``acf_y`` a series or a field; L is their length.
    """
    L = len(acf_x)
    weight = 1 - np.arange(1, L + 1) / L
    te = 1 + 2 * np.tensordot(weight * acf_x, acf_y, axes=(0, 0))
    return np.maximum(te, 1.0)


def effective_n(y, x, L: int = 50):
    """Effective sample size N_e = N / T_e of two autocorrelated series (Metz 1991)."""
    return len(y) / float(decorrelation_time(acf(y, L), acf(x, L)))


def t_factor(dof, ci_percent: float = 90):
    """Half-width of the confidence interval in units of the standard error."""
    return stats.t.ppf(0.5 + ci_percent / 200, dof)


# ---------------------------------------------------------------------------
# lagged regression on an index
# ---------------------------------------------------------------------------

def lag_pairs(y, x, lag):
    """y(t) next to x(t - lag), as two arrays of equal length."""
    if lag > 0:
        return y[lag:], x[:-lag]
    if lag < 0:
        return y[:lag], x[-lag:]
    return y, x


def lagged_regression(field, index, lags):
    """Regress ``field`` on the standardised ``index`` at several lags.

    field(t) is paired with index(t - lag): at a negative lag the field value
    comes from before the index year, at a positive lag from after it.
    Returns one array with a ``lag`` dimension, in field units per std of index.
    """
    maps = []
    for lag in lags:
        f, i = xr.align(field, index.shift(time=lag).dropna('time'), join='inner')
        maps.append(((f - f.mean('time')) * standardise(i)).mean('time'))
    return xr.concat(maps, dim=pd.Index(list(lags), name='lag'))


def lagged_curve(y, x, lags, L: int = 50, ci_percent: float = 90):
    """Lagged regression of the series y on the series x, with confidence interval.

    The slope is in units of y per std of x, the same as ``lagged_regression``.
    The interval uses the effective sample size after Metz (1991).
    """
    rows = []
    for lag in lags:
        yl, xl = lag_pairs(np.asarray(y), np.asarray(x), lag)
        r = np.corrcoef(yl, xl)[0, 1]
        slope = r * yl.std()
        n_eff = effective_n(yl, xl, L)
        half = t_factor(n_eff - 2, ci_percent) * yl.std() * np.sqrt((1 - r ** 2) / (n_eff - 2))
        rows.append({'lag': lag, 'slope': slope, 'low': slope - half, 'high': slope + half,
                     'n_eff': n_eff})
    return pd.DataFrame(rows).set_index('lag')


def regression_pvalues(field, index, lag, L: int = 50, acf_field=None):
    """Two-sided p-value at each grid point for field(t) against index(t - lag).

    Correlation r at each grid point, effective sample size N_e = N / T_e with
    T_e after Metz (1991) from the autocorrelation of the grid point and of the
    index, t = r * sqrt((N_e - 2) / (1 - r^2)) with N_e - 2 degrees of freedom.
    The autocorrelation of the field does not depend on the lag, so it can be
    computed once and passed as ``acf_field``.
    """
    f, i = xr.align(field, index.shift(time=lag).dropna('time'), join='inner')
    f = f.transpose('time', 'lat', 'lon')
    y = f.values - f.values.mean(axis=0)
    x = standardise(i).values.astype(y.dtype)
    n = len(x)
    r = np.tensordot(x, y, axes=(0, 0)) / np.sqrt(n * np.einsum('tij,tij->ij', y, y))

    if acf_field is None:
        acf_field = acf(field.transpose('time', 'lat', 'lon').values, L)
    te = decorrelation_time(acf_field, acf(index.values, len(acf_field)))
    n_eff = n / te
    t = r * np.sqrt((n_eff - 2) / (1 - r ** 2))
    p = 2 * stats.t.sf(np.abs(t), n_eff - 2)
    return f.isel(time=0, drop=True).copy(data=p.astype('float32'))


# ---------------------------------------------------------------------------
# composites of selected years
# ---------------------------------------------------------------------------

def event_numbers(selected):
    """Number of the event (run of consecutive years) each selected year belongs to."""
    s = np.asarray(selected)
    starts = s & ~np.r_[False, s[:-1]]
    return np.cumsum(starts)[s]


def count_events(selected):
    return int(event_numbers(selected).max()) if np.any(selected) else 0


def composite_mean_ci(y, selected, ci_percent: float = 90):
    """Mean of y over the selected years, with confidence interval.

    The interval comes from a one-sample t-test across events: each event
    contributes its own mean, so the sample size is the number of events.
    """
    y, selected = np.asarray(y), np.asarray(selected)
    event_means = pd.Series(y[selected]).groupby(event_numbers(selected)).mean()
    n = len(event_means)
    half = t_factor(n - 1, ci_percent) * event_means.std(ddof=1) / np.sqrt(n)
    mean = y[selected].mean()
    return mean, mean - half, mean + half


def composite_pvalues(field, selected):
    """Two-sided p-value at each grid point for a composite mean of ``field``.

    Neighbouring years of filtered data are nearly the same, so every run of
    consecutive selected years counts as one event. The mean over the selected
    years is compared with the standard error across events, with events - 1
    degrees of freedom (same test as ``composite_mean_ci``). ``field`` has the
    dimensions (time, lat, lon).
    """
    selected = np.asarray(selected)
    events = event_numbers(selected)
    values = field.values[selected]
    n = int(events.max())
    event_means = np.stack([values[events == k].mean(axis=0) for k in range(1, n + 1)])
    t = values.mean(axis=0) / (event_means.std(axis=0, ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    return field.isel(time=0, drop=True).copy(data=p.astype('float32'))


# ---------------------------------------------------------------------------
# significance on maps
# ---------------------------------------------------------------------------

def fdr_mask(p, alpha_fdr: float = 0.10):
    """Field significance: control of the false discovery rate (Wilks 2016).

    Benjamini-Hochberg: the p-values of all valid grid points are sorted, the
    largest p_(i) with p_(i) <= (i / N) * alpha_fdr is the threshold, and every
    grid point with p at or below it is significant. NaN points are ignored.
    """
    values = np.asarray(p)
    valid = np.isfinite(values)
    ranked = np.sort(values[valid])
    passed = ranked <= np.arange(1, ranked.size + 1) / ranked.size * alpha_fdr
    limit = ranked[passed].max() if passed.any() else -np.inf
    mask = valid & (values <= limit)
    return p.copy(data=mask) if isinstance(p, xr.DataArray) else mask


def stipple_mask(p, use_fdr: bool = True, alpha_fdr: float = 0.10, ci_percent: float = 90):
    """The mask that is stippled: FDR if ``use_fdr``, otherwise the local test."""
    return fdr_mask(p, alpha_fdr) if use_fdr else p < 1 - ci_percent / 100


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------

def to_points(da):
    """Field (time, lat, lon) -> array (time, points) of the grid points without NaN."""
    values = da.transpose('time', 'lat', 'lon').values.reshape(da.sizes['time'], -1)
    valid = np.isfinite(values).all(axis=0)
    return (values if valid.all() else values[:, valid]), valid


def to_map(values, valid, template):
    """One value per valid grid point -> (lat, lon) map on the grid of ``template``."""
    out = np.full(valid.shape, np.nan, dtype='float32')
    out[valid] = values
    return xr.DataArray(
        out.reshape(template.sizes['lat'], template.sizes['lon']), dims=('lat', 'lon'),
        coords={'lat': template['lat'], 'lon': template['lon']},
    )


def plot_map(ax, da, title, levels, cmap='RdBu_r', fontsize=10, sig=None, step: int = 4,
             stipple_size: float = 1.5, stipple_color='0.25',
             lat_limit: Optional[float] = None, label: Optional[str] = None):
    """Filled map with a short title.

    ``sig`` is an optional boolean (lat, lon) array; every ``step``-th grid
    point (in lat and lon) at which it is True gets a small grey dot.
    ``lat_limit`` crops the view to ±lat_limit degrees; the global mean is
    still taken over the whole globe. If ``label`` is given, the global mean
    of the map is written in the lower left corner behind it ('' = number only).
    """
    im = da.plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), levels=levels, cmap=cmap,
        extend='both', add_colorbar=False, add_labels=False,
    )
    if sig is not None:
        thinned = sig.transpose('lat', 'lon').isel(
            lat=slice(step // 2, None, step), lon=slice(step // 2, None, step))
        lon2d, lat2d = np.meshgrid(thinned['lon'].values, thinned['lat'].values)
        ax.scatter(lon2d[thinned.values], lat2d[thinned.values], s=stipple_size,
                   color=stipple_color, linewidths=0, transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.set_global()
    if lat_limit is not None:
        _, y_limit = ax.projection.transform_point(180, lat_limit, ccrs.PlateCarree())
        ax.set_ylim(-y_limit, y_limit)
    ax.set_title(title, fontsize=fontsize)
    if label is not None:
        ax.text(0.02, 0.05, f'{label}{float(global_mean(da)):+.3f}', transform=ax.transAxes,
                fontsize=fontsize, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))
    return im


def band_contributions(da, tropics_lat: float = 30) -> dict:
    """Contribution of three latitude bands to the area-weighted global mean of ``da``.

    Each band's weighted sum is divided by the weight of the whole (valid) globe,
    so the three parts add up to global_mean(da). 
    """
    w = area_weights(da['lat']).broadcast_like(da).where(da.notnull())
    total = w.sum()
    bands = {
        f'{tropics_lat}S-{tropics_lat}N': abs(da['lat']) <= tropics_lat,
        f'north of {tropics_lat}N': da['lat'] > tropics_lat,
        f'south of {tropics_lat}S': da['lat'] < -tropics_lat,
    }
    return {band: float((da * w).where(sel).sum() / total) for band, sel in bands.items()}


def pattern_correlation(a, b):
    """Area-weighted pattern correlation of two maps."""
    a, b = a.transpose('lat', 'lon'), b.transpose('lat', 'lon')
    weights = np.broadcast_to(np.cos(np.deg2rad(a['lat'].values))[:, None], a.shape)
    ok = np.isfinite(a.values) & np.isfinite(b.values)
    w, x, y = weights[ok], a.values[ok], b.values[ok]
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    return np.average(x * y, weights=w) / np.sqrt(np.average(x * x, weights=w) * np.average(y * y, weights=w))


# ---------------------------------------------------------------------------
# multiple linear regression with the Metz test
# ``predictors`` is a DataFrame of standardised predictors, one row per year.
# ---------------------------------------------------------------------------

def product_name(xcol, ycol):
    """Column name of the product of two predictors, e.g. 'PDVxAMV'."""
    return f'{xcol}x{ycol}'


def design(predictors: pd.DataFrame, terms):
    """Intercept plus the given predictor columns, as a (time, k + 1) array."""
    return np.column_stack([np.ones(len(predictors))] + [predictors[term].values for term in terms])


def residual(a, Z):
    """What is left of ``a`` after a least-squares fit on the columns of Z."""
    Z = Z.astype(a.dtype)
    return a - Z @ (np.linalg.pinv(Z) @ a)


def sum_of_squares(a):
    return np.einsum('t...,t...->...', a, a)


def r_squared(y, predictors: pd.DataFrame, terms):
    """Share of the variance of y explained by the model with ``terms``."""
    y = np.asarray(y)
    return 1 - sum_of_squares(residual(y, design(predictors, terms))) \
        / sum_of_squares(residual(y, design(predictors, [])))


def partial_regression(y, predictors: pd.DataFrame, terms, term, L: int = 50,
                       ci_percent: float = 90):
    """Coefficient of ``term`` in the MLR of y on ``terms``, with its t-test.

    The other predictors are removed from y and from the predictor first; the
    slope between the two residual series is the MLR coefficient. Test:
    N_e = N / T_e with T_e after Metz (1991), degrees of freedom N_e minus the
    number of fitted parameters (N_e - 2 for a single regression). y is a
    series (time,) or a field as (time, points).
    """
    others = design(predictors, [other for other in terms if other != term])
    y = residual(np.asarray(y), others)
    x = residual(predictors[term].values, others).astype(y.dtype)

    sxx, syy = sum_of_squares(x), sum_of_squares(y)
    coef = x @ y / sxx
    r = coef * np.sqrt(sxx / syy)

    n_eff = len(x) / decorrelation_time(acf(y, L), acf(x, L))
    dof = n_eff - len(terms) - 1
    se = np.sqrt(syy / sxx * (1 - r ** 2) / dof)
    return {
        'coef': coef, 'half': t_factor(dof, ci_percent) * se, 'n_eff': n_eff,
        'p': 2 * stats.t.sf(np.abs(coef) / se, dof),
    }


def fit_series(y, predictors: pd.DataFrame, terms, L: int = 50, ci_percent: float = 90):
    """One MLR of a series: a row per term with interval, p-value, N_e and R²."""
    r2 = float(r_squared(y, predictors, terms))
    rows = {}
    for term in terms:
        fit = partial_regression(y, predictors, terms, term, L, ci_percent)
        coef, half = float(fit['coef']), float(fit['half'])
        rows[term] = {'coef': coef, 'low': coef - half, 'high': coef + half,
                      'p': float(fit['p']), 'n_eff': float(fit['n_eff']), 'R2': r2}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# results on disk: computed once, read back while the settings match
# ---------------------------------------------------------------------------

def load_result(name: str, results_dir, settings: dict, recompute: bool = False):
    """Dataset ``<results_dir>/<name>.nc``, or None if it has to be computed.

    None if the file is missing, ``recompute`` is set, or any of ``settings``
    differs from what the file was written with (the names are printed).
    """
    path = pathlib.Path(results_dir) / f'{name}.nc'
    if recompute or not path.exists():
        return None
    ds = xr.load_dataset(path)
    changed = [k for k, v in settings.items() if ds.attrs.get(k) != v]
    if changed:
        print(f'{path.name}: written with other settings ({", ".join(changed)}), computing again')
        return None
    print(f'read {path}')
    return ds


def save_result(ds: xr.Dataset, name: str, results_dir, settings: dict, source: str):
    """Write ``ds`` to ``<results_dir>/<name>.nc`` with ``settings`` as attributes."""
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ds.attrs.update(settings, source=source)
    path = results_dir / f'{name}.nc'
    ds.to_netcdf(path)
    print(f'wrote {path}')


def stack(items: dict, dim: str):
    """Concatenate a dict {label: DataArray} along a new dimension ``dim``."""
    return xr.concat(list(items.values()), dim=pd.Index(list(items), name=dim))


def stack_store(store: dict, value_names, key_names) -> xr.Dataset:
    """Dict {key tuple: (DataArray, ...)} -> Dataset along a dimension 'map'.

    The parts of the keys become string coordinates along 'map' (netCDF has no
    MultiIndex), e.g. scenario, field and term of every coefficient map.
    """
    keys = list(store)
    data = {v: xr.concat([store[k][j].rename(v) for k in keys], dim='map')
            for j, v in enumerate(value_names)}
    coords = {n: ('map', [str(k[j]) for k in keys]) for j, n in enumerate(key_names)}
    return xr.Dataset(data).assign_coords(coords)

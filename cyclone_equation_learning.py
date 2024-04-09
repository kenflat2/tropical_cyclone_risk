import dask
import datetime
import numpy as np
import os
import xarray as xr
import time

import namelist
from intensity import coupled_fast, ocean
from thermo import calc_thermo
from track import env_wind
from wind import tc_wind
from util import basins, input, mat


# We have to replace the variables year, n_tracks, basin
#   year - integer (duh)
#   month - integer from 0 to 11
#   b - basin of track
#
def init_track_object(year, month, b):
    # Load thermodynamic and ocean variables.
    fn_th = calc_thermo.get_fn_thermo()
    ds = xr.open_dataset(fn_th)
    dt_year_start = datetime.datetime(year-1, 12, 31)
    dt_year_end = datetime.datetime(year, 12, 31)
    dt_bounds = input.convert_from_datetime(ds, [dt_year_start, dt_year_end])
    ds = ds.sel(time=slice(dt_bounds[0], dt_bounds[1])).load()
    lon = ds['lon'].data
    lat = ds['lat'].data
    mld = ocean.mld_climatology(year, basins.TC_Basin('GL'))
    strat = ocean.strat_climatology(year, basins.TC_Basin('GL'))    # Make sure latitude is increasing.
    vpot = ds['vmax'] * namelist.PI_reduc * np.sqrt(namelist.Ck / namelist.Cd)
    rh_mid = ds['rh_mid']
    chi = ds['chi']

    if (lat[0] - lat[1]) > 0:
        vpot = vpot.reindex({'lat': lat[::-1]})
        rh_mid = rh_mid.reindex({'lat': lat[::-1]})
        chi = chi.reindex({'lat': lat[::-1]})
        lat = lat[::-1]

    # Load the basin bounds and genesis points.
    basin_ids = np.array(sorted([k for k in namelist.basin_bounds if k != 'GL']))
    f_basins = {}
    for basin_id in basin_ids:
        ds_b = xr.open_dataset('land/%s.nc' % basin_id)
        basin_mask = ds_b['basin']
        f_basins[basin_id] = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)

    # In case basin is "GL", we load again.
    ds_b = xr.open_dataset('land/%s.nc' % b.basin_id)
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    b_bounds = b.get_bounds()

    # To randomly seed in both space and time, load data for each month in the year.
    cpl_fast = [0] * 12
    m_init_fx = [0] * 12
    n_seeds = np.zeros((len(basin_ids), 12))
    T_s = namelist.total_track_time_days * 24 * 60 * 60     # total time to run tracks
    fn_wnd_stat = env_wind.get_env_wnd_fn()
    ds_wnd = xr.open_dataset(fn_wnd_stat)
    month_i = month
    dt_month = datetime.datetime(year, month_i + 1, 15)
    ds_dt_month = input.convert_from_datetime(ds_wnd, [dt_month])[0]
    vpot_month = np.nan_to_num(vpot.interp(time = ds_dt_month).data, 0)
    rh_mid_month = rh_mid.interp(time = ds_dt_month).data
    chi_month = chi.interp(time = ds_dt_month).data
    chi_month[np.isnan(chi_month)] = 5
    m_init_fx = mat.interp2_fx(lon, lat, rh_mid_month)
    chi_month = np.maximum(np.minimum(np.exp(np.log(chi_month + 1e-3) + namelist.log_chi_fac) + namelist.chi_fac, 5), 1e-5)

    mld_month = mat.interp_2d_grid(mld['lon'], mld['lat'], np.nan_to_num(mld[:, :, month_i]), lon, lat)
    strat_month = mat.interp_2d_grid(strat['lon'], strat['lat'], np.nan_to_num(strat[:, :, month_i]), lon, lat)
    cpl_fast = coupled_fast.Coupled_FAST(fn_wnd_stat, b, ds_dt_month,
                                            namelist.output_interval_s, T_s)
    cpl_fast.init_fields(lon, lat, chi_month, vpot_month, mld_month, strat_month)

    return cpl_fast
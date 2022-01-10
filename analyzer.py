import os
import toml
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from glob import glob
from pathlib import Path

colors = {
    'car': 'blue',
    'truck': 'red',
}

mph_per_ms = 2.23694
kph_per_ms = 3.6

def get_track_time(fpath):
    chunks = fpath.name.split('_', 1)
    if len(chunks) == 0:
        return None
    else:
        stamp, _ = chunks
        return pd.to_datetime(stamp)

def date_range(min_time=None, max_time=None, path='tracks'):
    path = Path(path)
    fpaths = list(path.glob('*.csv'))
    ftimes = [get_track_time(fn) for fn in fpaths]
    if min_time is not None:
        min_time = pd.to_datetime(min_time)
        fpaths = [fn for fn, ft in zip(fpaths, ftimes) if ft >= min_time]
    if max_time is not None:
        max_time = pd.to_datetime(max_time)
        fpaths = [fn for fn, ft in zip(fpaths, ftimes) if ft <= max_time]
    return list(fpaths)

def load_track(path='tracks', norm=True):
    if type(path) is list:
        return {fp: load_track(fp, norm=norm) for fp in path}
    if os.path.isdir(path):
        return {fp: load_track(fp, norm=norm) for fp in glob('tracks/*.csv')}
    data = pd.read_csv(path)
    if norm:
        data['t'] -= data['t'].iloc[0]
    return data

def path_info(path):
    _, fname = os.path.split(path)
    name, _ = os.path.splitext(fname)
    ts, lab, num = name.split('_')
    return ts, lab, num

def plot_track(path='tracks', disp='speed', ax=None):
    if os.path.isdir(path):
        fpaths = glob('tracks/*.csv')
    else:
        fpaths = [path]

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()

    for fp in fpaths:
        ts, lab, num = path_info(fp)
        col = colors.get(lab, 'black')
        df = load_track(fp).set_index('t')

        if disp == 'path':
            df.plot.scatter(
                'x', 'y', color=col, xlim=(0, 1), ylim=(0, 1), s=5, ax=ax
            )
        elif disp == 'speed':
            df['x'].plot(
                ylim=(0, 1), color=col, marker='o', markersize=5, ax=ax
            )

# requires field of view info → fov: w x h
def calc_speed(data, fov, units='mph', rsquared=False):
    data = data.copy()
    data['t'] -= data['t'].iloc[0]
    data['x'] *= fov[0]
    data['y'] *= fov[1]

    res_x = smf.ols('x ~ 1 + t', data=data).fit()
    res_y = smf.ols('y ~ 1 + t', data=data).fit()

    vx = res_x.params['t']
    vy = res_y.params['t']

    σ2x = res_x.cov_params().loc['t', 't']
    σ2y = res_y.cov_params().loc['t', 't']

    v = np.sqrt(vx**2+vy**2)
    σ = np.sqrt((vx/v)**2*σ2x+(vy/v)**2*σ2y)

    if units == 'mph':
        v *= mph_per_ms
        σ *= mph_per_ms
    elif units == 'kph':
        v *= kph_per_ms
        σ *= kph_per_ms

    if rsquared:
        r2 = 0.5*(res_x.rsquared+res_y.rsquared)
        return v, σ, r2
    else:
        return v, σ

def track_info(path='tracks', data=None, fov='config.toml', units='mph'):
    if type(fov) is str:
        config = toml.load(fov)
        scene = config['scene']
        fov = scene['width'], scene['height']
    if data is None:
        data = load_track(path)
    if type(data) is dict:
        total = pd.DataFrame([
            track_info(path=fn, data=df, fov=fov, units=units) for fn, df in data.items()
        ])
        total['time'] = pd.to_datetime(total['time'])
        return total
    else:
        ts, lab, num = path_info(path)
        rang = data.max() - data.min()
        N, Δt = len(data), rang['t']
        Δx, Δy = rang['x'], rang['y']
        v, σ, r2 = calc_speed(data, fov, units=units, rsquared=True)
        return {
            'fname': path, 'time': ts, 'label': lab, 'number': num,
            'frames': N, 'Δt': Δt, 'Δx': Δx, 'Δy': Δy, 'v': v, 'σ': σ,
            'r2': r2,
        }

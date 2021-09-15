import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from glob import glob

colors = {
    'car': 'blue',
    'truck': 'red',
    'person': 'black',
}

mph_per_ms = 2.23694

def load_track(path, norm=True):
    if os.path.isdir(path):
        return {fp: load_track(fp, norm=norm) for fp in glob('tracks/*.csv')}
    data = pd.read_csv(path)
    data['t'] = data['t'] - data['t'][0]
    data['x'] = data['x'] - data['x'][0]
    data = data.set_index('t')
    return data

def plot_track(path='tracks', disp='speed', ax=None):
    if os.path.isdir(path):
        fpaths = glob('tracks/*.csv')
    else:
        fpaths = [path]

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()

    for fp in fpaths:
        _, fn = os.path.split(fp)
        name, _ = os.path.splitext(fn)
        lab, num, ts = name.split('_')
        if lab == 'person':
            continue

        col = colors.get(lab, 'black')
        df = load_track(fp)

        if disp == 'path':
            df.plot.scatter(
                'x', 'y', color=col, xlim=(0, 1), ylim=(0, 1), s=5, ax=ax
            )
        elif disp == 'speed':
            df['x'].plot(
                ylim=(-0.1, 1), color=col, marker='o', markersize=5, ax=ax
            )

def calc_speed(data, fov, units='imperial'):
    data = data.copy().reset_index().assign(one=1)
    data['x'] *= fov[0]
    data['y'] *= fov[1]

    res_x = smf.ols('x ~ 1 + t', data=data).fit()
    res_y = smf.ols('y ~ 1 + t', data=data).fit()

    vx = res_x.params['t']
    vy = res_y.params['t']

    σx = res_x.cov_params().loc['t', 't']
    σy = res_y.cov_params().loc['t', 't']

    v = np.sqrt(vx**2+vy**2)
    σ = np.sqrt(σx**2+σy**2)

    if units == 'imperial':
        v *= mph_per_ms
        σ *= mph_per_ms

    return v, σ

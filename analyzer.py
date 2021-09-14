import os
import numpy as np
import pandas as pd

from glob import glob

colors = {
    'car': 'blue',
    'truck': 'red',
    'person': 'black',
}

def load_track(path, norm=True):
    if os.path.isdir(path):
        return {fp: load_track(fp, norm=norm) for fp in glob('tracks/*.csv')}
    data = pd.read_csv(path)
    data['time'] = data['time'] - data['time'][0]
    data['x'] = data['x'] - data['x'][0]
    data = data.set_index('time')
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

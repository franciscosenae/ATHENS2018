import scipy.io
from scipy.stats import skew
from scipy.fftpack import fft
import numpy as np
import pandas as pd


ACTIVITIES = ['brushing', 'drinking', 'shoe', 'writing']
FREQUENCY = 128


def means(data):
    gx = []
    gy = []
    gz = []
    for ac in data.keys():
        for s in data[ac]:
            gx.append(s['x'].mean())
            gy.append(s['y'].mean())
            gz.append(s['z'].mean())
    return gx, gy, gz


def get_std(data):
    std = []
    for ac in data.keys():
        for s in data[ac]:
            r = dimension_reduction(s)
            std.append(r.std())
    return std


def get_skewness(data):
    out = []
    for ac in data.keys():
        for s in data[ac]:
            r = dimension_reduction(s)
            out.append(skew(r)[0])
    return out


def get_labels(data):
    labels = []
    for ac in data.keys():
        for s in data[ac]:
            labels.append(ac)
    return labels


def dimension_reduction(single_person_activity):
    return (np.sqrt(single_person_activity['x']**2
                    + single_person_activity['y']**2
                    + single_person_activity['z']**2))


def remove_dc(data):
    out = data
    for ac in data.keys():
        for s in data[ac]:
            s['x'] -= s['x'].mean()
            s['y'] -= s['y'].mean()
            s['z'] -= s['z'].mean()
    return out


def energy25_75(r, freq):
    N = len(r)
    R = np.abs(fft(r.flatten()))**2
    R[0] = 0
    frequencies = [i * freq / N for i in range(N)]
    CR = R.cumsum()
    CR /= CR.max()

    index = np.where(CR < 0.25/2)[0][-1]
    f25 = frequencies[index]

    index = np.where(CR < 0.75/2)[0][-1]
    f75 = frequencies[index]

    return f25, f75


def get_energy(data):
    f25 = []
    f75 = []
    for ac in data.keys():
        for s in data[ac]:
            r = dimension_reduction(s)
            r = r-r.mean()
            _f25, _f75 = energy25_75(r, FREQUENCY)
            f25.append(_f25)
            f75.append(_f75)
    return f25, f75


def format_data(data):
    out = {}
    for label in ACTIVITIES:
        out[label] = []
        activity_data = data['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            out[label].append({
                'x': activity_data[0,i]['x'],
                'y': activity_data[0,i]['y'],
                'z': activity_data[0,i]['z'],
            })
    return out


def load_dataframe(filename):
    data = scipy.io.loadmat(filename)
    acnames = data['data'].dtype.names
    print(acnames)
    data['data'].dtype.names = [
        n if n!='shoelacing' else 'shoe' for n in data['data'].dtype.names
    ]

    data = format_data(data)

    gx, gy, gz = means(data)
    labels = get_labels(data)
    std = get_std(data)
    skewness = get_skewness(data)
    f25, f75 = get_energy(data)


    df = pd.DataFrame({
        'gx': gx,
        'gy': gy,
        'gz': gz,
        'std': std,
        'skewness': skewness,
        'f25': f25,
        'f75': f75,
        'label': labels,
    })
    return df


def load_testdata(path='data/raw_from_matlab/testData.mat'):
    data = scipy.io.loadmat(path)
    _names = data['data'].dtype.names
    assert _names == ('x', 'y', 'z', 'Label')

    out = {}
    out['x'] = data['data']['x'][0, 0].flatten()
    out['y'] = data['data']['y'][0, 0].flatten()
    out['z'] = data['data']['z'][0, 0].flatten()
    out['label'] = data['data']['Label'][0, 0].flatten()
    return pd.DataFrame(out)


def make_windowed(data, size=20, steps=2, freq=128):
    """Size and steps in seconds"""
    _window_start, _window_end = 0, size*128
    while _window_end < len(testdata):
        data.iloc[_window_start:_window_end]

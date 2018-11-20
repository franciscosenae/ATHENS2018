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
    for label in ACTIVITIES:
        activity_data = data['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            gx.append(activity_data[0,i]['x'].mean())
            gy.append(activity_data[0,i]['y'].mean())
            gz.append(activity_data[0,i]['z'].mean())
    return gx, gy, gz


def get_std(data):
    std = []
    for label in ACTIVITIES:
        activity_data = data['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            r = dimension_reduction(activity_data[0,i])
            std.append(r.std())
    return std


def get_skewness(data):
    out = []
    for label in ACTIVITIES:
        activity_data = data['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            r = dimension_reduction(activity_data[0,i])
            out.append(skew(r)[0])
    return out


def get_labels(data):
    labels = []
    for label in ACTIVITIES:
        activity_data = data['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            labels.append(label)
    return labels


def dimension_reduction(single_person_activity):
    return (np.sqrt(single_person_activity['x']**2
                    + single_person_activity['y']**2
                    + single_person_activity['z']**2))


def remove_dc(data):
    out = data
    for label in ACTIVITIES:
        activity_data = out['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            activity_data[0,i]['x'] -= activity_data[0,i]['x'].mean()
            activity_data[0,i]['y'] -= activity_data[0,i]['y'].mean()
            activity_data[0,i]['z'] -= activity_data[0,i]['z'].mean()
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
    for label in ACTIVITIES:
        activity_data = data['data'][label][0,0]
        for i in range(activity_data.shape[1]):
            r = dimension_reduction(activity_data[0,i])
            r = r-r.mean()
            _f25, _f75 = energy25_75(r, FREQUENCY)
            f25.append(_f25)
            f75.append(_f75)
    return f25, f75


def load_dataframe(filename):
    data = scipy.io.loadmat(filename)
    acnames = data['data'].dtype.names
    data['data'].dtype.names = [
        n if n!='shoelacing' else 'shoe' for n in data['data'].dtype.names
    ]

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


def main():
    _df1 = load_dataframe('data/raw_from_matlab/data2016.mat')
    _df1['brushing'] = _df1.label == 'brushing'
    _df2 = load_dataframe('data/raw_from_matlab/data2017.mat')
    _df2['brushing'] = _df2.label == 'brushing'
    _df3 = load_dataframe('data/raw_from_matlab/data2018.mat')
    _df3['brushing'] = _df3.label == 'brushing'

    df = pd.concat((_df1, _df2, _df3))

    _df1.to_csv('2016.csv')
    _df2.to_csv('2017.csv')
    _df3.to_csv('2018.csv')
    df.to_csv('all.csv')



if __name__ == '__main__':
    main()

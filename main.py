import scipy.io
from scipy.stats import skew
from scipy.fftpack import fft
import numpy as np
import pandas as pd


ACTIVITIES = ['brushing', 'drinking', 'shoe', 'writing']
FREQUENCY = 128


def load_data(filename):
    data_2016 = scipy.io.loadmat(filename)
    mdata = data_2016["data"]
    activities = mdata.dtype.names
    # mdata[brushing][0][0] holds 13 elements
    ndata = {n: mdata[n][0, 0] for n in activities}
    for k in ndata.keys():
        ds_type = ndata[k].dtype
        ndata[k] = {n: ndata[k][n][0, 0] for n in ds_type.names}
    return ndata


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
    data = scipy.io.loadmat('../data2016.mat')

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
        'labels': labels,
    })
    return df


def main():
    df = load_dataframe('../data2016.mat')




if __name__ == '__main__':
    main()

import scipy.io
from scipy.stats import skew
from scipy.fftpack import fft
import numpy as np
import pandas as pd
import tqdm


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
            out.append(skew(r))
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
    R = np.abs(fft(r))**2
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
                'x': activity_data[0,i]['x'].flatten(),
                'y': activity_data[0,i]['y'].flatten(),
                'z': activity_data[0,i]['z'].flatten(),
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
        'gx_abs': np.abs(gx),
        'gy_abs': np.abs(gy),
        'gz_abs': np.abs(gz),
    })
    return df


def create_data_for_svm(data: dict):
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
        'gx_abs': np.abs(gx),
        'gy_abs': np.abs(gy),
        'gz_abs': np.abs(gz),
    })

    if 'unknown' in data.keys():
        df['window_start'] = [a['window_start'] for a in data['unknown']]
        df['window_end'] = [a['window_end'] for a in data['unknown']]
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

    _activities = ('undefined', 'drinking', 'brushing', 'writing', 'shoe')
    out['activity'] = [_activities[i] for i in out['label']]

    return pd.DataFrame(out)


def make_windowed(testdata, size=20, steps=2, freq=128):
    """Size and steps in seconds"""
    out = {'unknown': []}
    _window_start, _window_end = 0, size*128
    while _window_end < len(testdata):
        df_chunk = testdata.iloc[_window_start:_window_end]
        out['unknown'].append({
            'x': df_chunk['x'],
            'y': df_chunk['y'],
            'z': df_chunk['z'],
            'window_start': _window_start,
            'window_end': _window_end,
        })
        _window_start += steps*freq
        _window_end += steps*freq
    return out


def combine_predictions(test_data, test_df, window_predictions):
    # decision_function_values = clf.decision_function(X)
    predictions = window_predictions

    times = list(test_data.index)
    times_predictions = []
    # times_predictions2 = []
    for i in tqdm.tqdm(times):
        relevant_predictions = predictions[(test_df.window_start <= i) & (test_df.window_end > i)]
        # relevant_function_value = predictions[(test_df.window_start <= i) & (test_df.window_end > i)]
        final_prediction = bool(round(relevant_predictions.mean()))
        # final_prediction2 = relevant_function_value.mean() > 0
        times_predictions.append(final_prediction)
        # times_predictions2.append(final_prediction2)
    return times_predictions


def predict_on_streamed_data(clf, test_data, features, window_size=20, window_step=2):
    test_df = create_data_for_svm(make_windowed(test_data, size=20, steps=2))
    # test_df.sample(5)

    X = test_df[features]
    window_predictions = clf.predict(X)

    time_predictions = combine_predictions(test_data, test_df, window_predictions)

    return time_predictions

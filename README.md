# ATHENS 2018 - Health Informatics
Goal of this assignment was to recognize *Activities of Daily Living* (ADLs)
from sensor data.

## About the task, sensor, data, and method
### Sensor
The sensor used was the
[Shimmer 3](http://www.shimmersensing.com/products/shimmer3-imu-sensor)
(or similar). It is capable of measuring acceleration along three axes.

### Raw Data
For data collection, each student performed four activities:
- Brushing teeth (20 seconds)
- Binding shoes (normal speed)
- Drinking water (three movements)
- Writing (a rather long sentence)
The labeled data can be found in `data/raw_from_matlab/data2018.mat`.
This file therefore contains a number of measurements, each representing a
single activity.

### Feature Extraction
In order to train a classifier on this data, we extracted features from each
measurement, such as:
- The mean acceleration along each axis (`gx`, `gy`, `gz`)
- The standard deviation and skewness of the overall acceleration (`std`, `skewness`)
- The 25. and 75. percentile of the Fourier transform of the measurements
  (`f25`, `f75`)

### Classifier
We then trained a support vector machine (SVM) on on these features. As can be
seen in `Assignment.ipynb`, the collected data can be perfectly separated with
just two features and a linear classifier.

### Test Data
In a realistic setting, the task would be very different: The data would consist
of one continuous stream of accelerometer data, and each point in time then has
to be classified. We therefore can not directly apply the above method.

The dataset that was provided to us can be found in
`data/raw_from_matlab/testData.mat**.
By using sliding windows, we generate chunks of data, for which we can then
again extract the features and use those to classify this chunk.
Finally, the different predicted labels have to be combined in order to generate
a single label for each point in time. Majority voting worked well for this task.


## Setup
Install all necessary packages, ideally in a
[virtual environment](https://docs.python.org/3/library/venv.html)
```pip install -r requirements.txt```


## Usage
The jupyter notebook `Assignment.ipynb` provides all the function calls and goes
through the described approach step by step.
Open jupyter:
```jupyter notebook```
Then go through `Assignment.ipynb` in your browser.

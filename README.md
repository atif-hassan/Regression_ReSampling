[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/reg-resampler.svg)](https://pypi.python.org/pypi/reg-resampler/)
[![Downloads](https://pepy.tech/badge/reg-resampler)](https://pepy.tech/project/reg-resampler)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/Regression_ReSampling/commits/master)
# Regression ReSampling
An interface to apply your favourite resampler on regression tasks. Currently supports all resampling techniques present in **imblearn**

## Why does this exist?
While we were working on a regression task, we realized that the target variable was skewed, i.e., most samples were present in a particular range. One can easily solve the skew problem for classification tasks via a slew of resampling techniques (either under or over sampling) but this luxury is unavailable for regression tasks. We therefore decided to create an interface that can repurpose all resampling techniques for classification problems to regression problems! 

## How to install?
```pip install reg_resampler```

### Functions and parameters
```python
# This returns a numpy list of classes for each corresponding sample. It also automatically merges classes when required
fit(X, target, bins=3, min_n_samples=6, balanced_binning=False, verbose=2)
```
#### Parameters:
- **X** - Either a pandas dataframe or numpy matrix. Complete data to be resampled.
- **target** - Either string (for pandas) or index (for numpy). The target variable to be resampled.
- **bins=3** - The number of classes that the user wants to generate. (Default: 3)
- **min_n_samples=6** - Number of minimum samples in each bin. Has to be more than neighbours in imblearn. (Default: 6)
- **balanced_binning=False** - Decides whether samples are to be distributed roughly equally across all classes. (Default: False)
- **verbose=2** - 0 will disable print by package, 1 will print info about class mergers and 2 will also print class distributions.

```python
# Performs resampling and returns the resampled dataframe/numpy matrices in the form of data and target variable.
resample(sampler_obj, trainX, trainY)
```
#### Parameters:
- **sampler_obj** - Your favourite resampling algorithm's object (currently supports imblearn)
- **trainX** - Either a pandas dataframe or numpy matrix. Data to be resampled. Also, contains the target variable
- **trainY** - Numpy array of psuedo classes obtained from fit function.

### Important Note
All functions return the same data type as provided in input.

### How to import?
```python
from reg_resampler import resampler
```

### How to use?
```python
# Initialize the resampler object
rs = resampler()

# You might recieve info about class merger for low sample classes
# Generate classes
Y_classes = rs.fit(df_train, target=target, bins=num_bins)
# Create the actual target variable
Y = df_train[target]

# Create a smote (over-sampling) object from imblearn
smote = SMOTE(random_state=27)

# Now resample
final_X, final_Y = rs.resample(smote, df_train, Y_classes)
```

## Tutorials
You can find further [tutorials](https://github.com/atif-hassan/Regression_ReSampling/tree/master/tutorials) on how to use this library for cross-validation

## Future Ideas
- Support for more resampling techniques

## Feature Request
Drop us an email at **atif.hit.hassan@gmail.com** or **pvsaikrithik@gmail.com** if you want any particular feature
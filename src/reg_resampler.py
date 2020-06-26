class resampler:
    def __init__(self):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from collections import Counter
        import numpy as np
        self.bins = 3
        self.pd = pd
        self.LabelEncoder = LabelEncoder
        self.Counter = Counter
        self.X = 0
        self.Y_classes = 0
        self.target = 0
        self.np = np

    # This function adds classes to each sample and returns the class list as a dataframe/numpy array (as per input)
    # It also merges classes as and when required
    def fit(self, X, target, bins=3, min_n_samples=6, balanced_binning=False, verbose=2):
        self.bins = bins
        tmp = target
        
        # If data is numpy, then convert it into pandas
        if type(target) == int:
            if target < 0:
                target = X.shape[1]+target
                tmp = target
            self.X = self.pd.DataFrame()
            for i in range(X.shape[1]):
                if i!=target:
                    self.X[str(i)] = X[:,i]
            self.X["target"] = X[:,target]
            target = "target"
        else:
            self.X = X.copy()
        
        # Use qcut if balanced binning is required
        if balanced_binning:
            self.Y_classes = self.pd.qcut(self.X[target], q=self.bins, precision=0)
        else:
            self.Y_classes = self.pd.cut(self.X[target], bins=self.bins)
        
        # Pandas outputs ranges after binning. Convert ranges to classes
        le = self.LabelEncoder()
        self.Y_classes = le.fit_transform(self.Y_classes)
        
        # Merge classes if number of neighbours is more than the number of samples
        classes_count = list(map(list, self.Counter(self.Y_classes).items()))
        classes_count = sorted(classes_count, key = lambda x: x[0])
        mid_point = len(classes_count)
        # Logic for merging
        for i in range(len(classes_count)):
              if classes_count[i][1] < min_n_samples:
                    self.Y_classes[self.np.where(self.Y_classes == classes_count[i][0])[0]] = classes_count[i-1][0]
                    if verbose > 0:
                        print("INFO: Class " + str(classes_count[i][0]) + " has been merged into Class " + str(classes_count[i-1][0]) + " due to low number of samples")
                    classes_count[i][0] = classes_count[i-1][0]
        if verbose > 0:
            print()
        
        # Perform label-encoding once again
        # Avoids class skipping after merging
        le = self.LabelEncoder()
        self.Y_classes = le.fit_transform(self.Y_classes)
        
        # Pretty print
        if verbose > 1:
            print("Class Distribution:\n-------------------")
            classes_count = list(map(list, self.Counter(self.Y_classes).items()))
            classes_count = sorted(classes_count, key = lambda x: x[0])
            for class_, count in classes_count:
                print(str(class_)+": "+str(count))
            print()
        
        # Finally concatenate and return as dataframe or numpy
        # Based on what type of target was sent
        self.X["classes"] = self.Y_classes
        if type(tmp) == int:
            self.target = tmp
        else:
            self.target = target
        return self.Y_classes 


    
    # This function performs the re-sampling
    def resample(self, sampler_obj, trainX, trainY):
        # If classes haven't yet been created, then run the "fit" function
        if type(self.Y_classes) == int:
            print("Error! Run fit method first!!")
            return None

        # Finally, perform the re-sampling
        resampled_data, _ = sampler_obj.fit_resample(trainX, trainY)
        if type(resampled_data).__module__ == 'numpy':
            resampled_data = self.pd.DataFrame(resampled_data, columns=self.X.drop("classes", axis=1).columns)

        # Return the correct X and Y
        if type(self.target) == int:
            return resampled_data.drop("target", axis=1).values, resampled_data["target"].values
        else:
            return resampled_data.drop(self.target, axis=1), resampled_data[self.target]

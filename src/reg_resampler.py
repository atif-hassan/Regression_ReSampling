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

    # This function adds classes to each sample and returns an augmented dataframe/numpy matrix
    def fit(self, X, target, bins=3, balanced_binning=False):
        self.bins = bins
        tmp = target
        
        # If data is numpy, then convert it into pandas
        if type(target) == int:
            if target == -1:
                target = X.shape[1]-1
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
        
        # Pretty print
        print("Class Distribution:\n-------------------")
        classes_count = list(map(list, self.Counter(self.Y_classes).items()))
        classes_count = sorted(classes_count, key = lambda x: x[0])
        for class_, count in classes_count:
            print(str(class_)+": "+str(count))
        
        # Finally concatenate and return as dataframe or numpy
        # Based on what type of target was sent
        self.X["classes"] = self.Y_classes
        if type(tmp) == int:
            self.target = tmp
            return self.X.values
        else:
            self.target = target
            return self.X
        
    # This function performs the re-sampling
    # It also merges classes as and when required
    def resample(self, sampler_obj):
        # If classes haven't yet been created, then run the "fit" function
        if type(self.Y_classes) == int:
            print("Error! Run fit method first!!")
            return None

        # These are the imblearn parameters that require certain number of samples
        # So we need to merge classes having samples less than the value of these hyper-parameters
        k_nbs, n_nbs, m_nbs, n_nbs_v3 = 0, 0, 0, 0
        params = sampler_obj.get_params()
        
        if "k_neighbors" in params:
            k_nbs = params["k_neighbors"]
        if "n_neighbors" in params:
            n_nbs = params["n_neighbors"]
        if "n_neighbors_ver3" in params:
            n_nbs_v3 = params["n_neighbors_ver3"]
        if "m_neighbors" in params:
            m_nbs = params["m_neighbors"]
        if "smote__k_neighbors" in params:
            k_nbs = params["smote__k_neighbors"]

        # Choose the max value
        nbs = max([k_nbs, n_nbs, m_nbs, n_nbs_v3])

        # Merge classes if number of neighbours is more than the number of samples
        if nbs > 0:
            classes_count = list(map(list, self.Counter(self.Y_classes).items()))
            classes_count = sorted(classes_count, key = lambda x: x[0])
            mid_point = len(classes_count)//2
            # Logic for merging
            for i in range(len(classes_count)):
                if i <= mid_point:
                    if classes_count[i][1] <= nbs:
                        self.Y_classes[self.np.where(self.Y_classes == classes_count[i][0])[0]] = classes_count[i+1][0]
                        print("Warning: Class " + str(classes_count[i][0]) + " has been merged into Class " + str(classes_count[i+1][0]) + " due to low number of samples")
                        classes_count[i][0] = classes_count[i+1][0]
                else:
                    if classes_count[i][1] <= nbs:
                        self.Y_classes[self.np.where(self.Y_classes == classes_count[i][0])[0]] = classes_count[i-1][0]
                        print("Warning: Class " + str(classes_count[i][0]) + " has been merged into Class " + str(classes_count[i-1][0]) + " due to low number of samples")
                        classes_count[i][0] = classes_count[i-1][0]

        # Finally, perform the re-sampling
        resampled_data, _ = sampler_obj.fit_resample(self.X, self.Y_classes)
        # Drop the extra class
        resampled_data.drop(["classes"], axis=1, inplace=True)
        # Return the correct type
        if type(self.target) == int:
            return resampled_data.values
        else:
            return resampled_data

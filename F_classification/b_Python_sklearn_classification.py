import sys
sys.path.append("../")

import os
import pandas as pd
import numpy as np
from joblib import load

from C_scrub.b_Python_Pandas_data_engineering import scrub_feature_engineering

#%% System setup

BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'A_data')
RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'b_Python_sklearn')
RESULTS_DIR = os.path.join(BASE_DIR, 'G_results')

#%% Classify

data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
id = data.pop('id')
data = scrub_feature_engineering(data)

finalModel = load(os.path.join(RESOURCES_DIR, 'classifierXGBoost_saved.pkl'))
outcome = finalModel.predict(data)

mapping = {0: 'lived', 1: 'euthanized', 2: 'died'}
outcome = np.vectorize(mapping.get)(outcome)


#%% Create file

output = pd.DataFrame({'id': id, 'outcome': outcome})
output.to_csv(os.path.join(RESULTS_DIR, 'b_Python_sklearn_output.csv'), index=False)

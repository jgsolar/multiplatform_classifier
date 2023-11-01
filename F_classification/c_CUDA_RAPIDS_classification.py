import sys
sys.path.append("../")

import os
import cudf
import numpy as np
import pandas as pd
from joblib import load

from C_scrub.c_CUDA_RAPIDS_data_engineering import scrub_feature_engineering

#%% System setup

BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'A_data')
RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'c_CUDA_RAPIDS')
RESULTS_DIR = os.path.join(BASE_DIR, 'G_results')

#%% Classify

data = cudf.read_csv(os.path.join(DATA_DIR, "test.csv"))
id = data.pop('id')

data = scrub_feature_engineering(data, train=False)

finalModel = load(os.path.join(RESOURCES_DIR, 'model_xgboost_complete_v0.1.joblib'))
outcome = finalModel.predict(data)

mapping = {0: 'lived', 1: 'euthanized', 2: 'died'}
outcome = np.vectorize(mapping.get)(outcome)


#%% Create file

id = id.to_numpy()
output = pd.DataFrame({'id': id, 'outcome': outcome})
output.to_csv(os.path.join(RESULTS_DIR , 'c_CUDA_RAPIDS_output.csv'), index=False)

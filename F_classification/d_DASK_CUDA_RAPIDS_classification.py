import sys
sys.path.append("../")

import os
import cudf
import numpy as np
import pandas as pd
import xgboost

from joblib import load

from C_scrub.d_DASK_CUDA_RAPIDS_data_engineering import scrub_feature_engineering

import dask
import dask_cudf
import dask.dataframe as dd
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

import distributed

#%% System setup

BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'A_data')
RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'd_DASK_CUDA_RAPIDS')
RESULTS_DIR = os.path.join(BASE_DIR, 'G_results')

dask.config.set({"array.backend": "cupy"})
dask.config.set({"dataframe.backend": "cudf"})

#%%

if __name__ == "__main__":
    def get_cluster():
        cluster = LocalCUDACluster(
            device_memory_limit='10GB',
            jit_unspill=True
        )
        client = Client(cluster)
        return client

    client = get_cluster()
    n_workers = len(client.scheduler_info()["workers"])



#%% Classify

data = dask_cudf.read_csv(os.path.join(DATA_DIR, "test.csv"))
data = scrub_feature_engineering(data, train=False)

data = dask_cudf.from_cudf(data, npartitions=2)
data_xgb = xgboost.dask.DaskQuantileDMatrix(client, data)

#%%

finalModel = load(os.path.join(RESOURCES_DIR, 'model_xgboost_dask_cuml_v0.1.joblib'))
finalModel.set_param({'predictor': 'gpu_predictor'})

outcome = xgboost.dask.predict(client, finalModel, data_xgb).compute()

#%%

mapping = {0: 'lived', 1: 'euthanized', 2: 'died'}
outcome = np.vectorize(mapping.get)(outcome)


#%% Create file

id = data.index.compute()
id = id.to_numpy()
output = pd.DataFrame({'id': id, 'outcome': outcome})
output.to_csv(os.path.join(RESULTS_DIR, 'd_DASK_CUDA_RAPIDS_output.csv'), index=False)

#%%

client.shutdown()

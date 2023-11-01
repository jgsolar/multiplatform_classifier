import sys

import cudf
import dask_cudf

sys.path.append("../")

import os
import dask

import dask.dataframe as dd
import cupy as cp

from joblib import dump, load

from B_resources.d_DASK_CUDA_RAPIDS.lesion_encoder import (code_lesion_site, code_lesion_type, code_lesion_sub_type,
                                                           code_lesion_code, lesion_number)

from cuml.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

#%% train

# import distributed
# from dask.distributed import Client, wait
# from dask_cuda import LocalCUDACluster
#
# dask.config.set({"array.backend": "cupy"})
# dask.config.set({"dataframe.backend": "cudf"})
#
# BASE_DIR = os.path.dirname(os.getcwd())
# DATA_DIR = os.path.join(BASE_DIR, 'A_data')
# RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'c_DASK_CUDA_RAPIDS')

#%%

# if __name__ == "__main__":
#     def get_cluster():
#         cluster = LocalCUDACluster(
#             device_memory_limit='10GB',
#             jit_unspill=True
#         )
#         client = Client(cluster)
#         return client
#
#     client = get_cluster()
#     n_workers = len(client.scheduler_info()["workers"])

#%%

# data_compet = dd.read_csv(os.path.join(DATA_DIR, "train.csv")).set_index('id')
# data_orig = dd.read_csv(os.path.join(DATA_DIR, 'horse.csv'))  # Include public data
#
# data_orig['id'] = data_orig.index + data_compet.index.max() + 1
# data_orig = data_orig.set_index('id')
#
# data = dd.concat([data_compet, data_orig])
# data = data.dropna(subset=['outcome'])
# data = data.drop_duplicates()
#
# del data_compet, data_orig
# train = True

#%% test

#
# import dask
# import dask_cudf
# import dask.dataframe as dd
# from dask.distributed import Client
# from dask_cuda import LocalCUDACluster
#
#
# BASE_DIR = os.path.dirname(os.getcwd())
# DATA_DIR = os.path.join(BASE_DIR, 'A_data')
# RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'd_DASK_CUDA_RAPIDS')
# RESULTS_DIR = os.path.join(BASE_DIR, 'G_results')
#
# dask.config.set({"array.backend": "cupy"})
# dask.config.set({"dataframe.backend": "cudf"})

#%%

# if __name__ == "__main__":
#     def get_cluster():
#         cluster = LocalCUDACluster(
#             device_memory_limit='10GB',
#             jit_unspill=True
#         )
#         client = Client(cluster)
#         return client
#
#     client = get_cluster()
#     n_workers = len(client.scheduler_info()["workers"])

#%%

# data = dd.read_csv(os.path.join(DATA_DIR, "test.csv"))
# train=False




# %% Clean data

def scrub_feature_engineering(data, train=False):

    if data.index.compute().name != 'id':
        data = data.reset_index(drop=True)
        data = data.set_index('id')

    #system setup
    BASE_DIR = os.path.dirname(os.getcwd())
    RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'd_DASK_CUDA_RAPIDS')

    # Drop cp_data, without information
    data = data.drop(columns=['cp_data'])

    # metadata definition
    data[["hospital_number", "lesion_1", "lesion_2", "lesion_3"]] = (
        data[["hospital_number", "lesion_1", "lesion_2", "lesion_3"]].astype(str))
    ohe_cols = data.select_dtypes(include=['object']).columns.to_list()
    if train:
        ohe_cols = [item for item in ohe_cols if item not in ['outcome']]
    num_cols = data.select_dtypes(include=['number']).columns.to_list()
    data[["hospital_number", "lesion_1", "lesion_2", "lesion_3"]] = (
        data[["hospital_number", "lesion_1", "lesion_2", "lesion_3"]].astype('int32'))

    # Replace unexpected values
    data['abdominal_distention'] = data['abdominal_distention'].replace('None', 'none')
    data['nasogastric_tube'] = data['nasogastric_tube'].replace('None', 'none')
    data['nasogastric_reflux'] = data['nasogastric_reflux'].replace('None', 'none')
    data['temp_of_extremities'] = data['temp_of_extremities'].replace('None', 'normal')
    data['peripheral_pulse'] = data['peripheral_pulse'].replace('None', 'normal')
    data['mucous_membrane'] = data['mucous_membrane'].replace('None', 'normal_pink')
    data['capillary_refill_time'] = data['capillary_refill_time'].replace('None', 'less_3_sec')
    data['capillary_refill_time'] = data['capillary_refill_time'].replace('3', 'less_3_sec')
    data['pain'] = data['pain'].replace('None', 'alert')
    data['pain'] = data['pain'].replace('moderate', 'mild_pain')
    data["pain"] = data["pain"].replace('slight', 'moderate')
    data['peristalsis'] = data['peristalsis'].replace('distend_small', 'normal')
    data['peristalsis'] = data['peristalsis'].replace('None', 'normal')
    data['nasogastric_reflux'] = data['nasogastric_reflux'].replace('slight', 'none')
    data['rectal_exam_feces'] = data['rectal_exam_feces'].replace('None', 'normal')
    data['rectal_exam_feces'] = data['rectal_exam_feces'].replace('serosanguious', 'normal')
    data['abdomen'] = data['abdomen'].replace('None', 'normal')
    data['abdomo_appearance'] = data['abdomo_appearance'].replace('None', 'clear')

    #%% Function - Lesion Treatment

    data['lesionSite'] = data['lesion_1'].map_partitions(code_lesion_site, meta=('lesionSite', 'int32'))
    data['lesionType'] = data['lesion_1'].map_partitions(code_lesion_type, meta=('lesionType', 'int32'))
    data['lesionSubType'] = data['lesion_1'].map_partitions(code_lesion_sub_type, meta=('lesionSubType', 'int32'))
    data['lesionCode'] = data['lesion_1'].map_partitions(code_lesion_code, meta=('lesionCode', 'int32'))
    data['numLesions'] = data.map_partitions(lesion_number, meta=('numLesions', 'int32'))

    data = data.drop(columns=['lesion_1', 'lesion_2', 'lesion_3'])
    ohe_cols = [item for item in ohe_cols if item not in ['lesion_1', 'lesion_2', 'lesion_3']]
    ohe_cols = ohe_cols + ['lesionSite', 'lesionType', 'lesionSubType', 'lesionCode']
    num_cols = num_cols + ['numLesions']

    # %% Coding hospital numbers

    if train:
        hospital_number_codifier = cudf.DataFrame(data['hospital_number'].value_counts().reset_index().compute())
        hospital_number_codifier.columns = ["hospital_number", "freq"]
        hospital_number_codifier = hospital_number_codifier.sort_values(by="freq", ascending=False)
    else:
        hospital_number_codifier = load(os.path.join(RESOURCES_DIR, 'hospital_number_codifier_v0.1.joblib'))
        mask = ~data['hospital_number'].isin(hospital_number_codifier['hospital_number'])
        most_frequent = hospital_number_codifier['hospital_number'].iloc[0]
        data['hospital_number'] = data['hospital_number'].mask(mask, most_frequent)
        del mask, most_frequent

    # %% Coding categorical variables - remove strings - keep dummies

    data['surgery'] = data["surgery"].map({'no': '0', 'yes': '1'}).astype('int32')
    data["age"] = data["age"].map({'young': '0', 'adult': '1'}).astype('int32')
    data["temp_of_extremities"] = data["temp_of_extremities"].map(
        {'cold': '0', 'cool': '1', 'normal': '2', 'warm': '3'}).astype('int32')
    data["peripheral_pulse"] = data["peripheral_pulse"].map({'absent': '0', 'reduced': '1', 'normal': '2',
                                                             'increased': '3'}).astype('int32')
    data["mucous_membrane"] = data["mucous_membrane"].map({'normal_pink': '0', 'bright_pink': '1', 'pale_pink': '2',
                                                           'pale_cyanotic': '3', 'bright_red': '4',
                                                           'dark_cyanotic': '5'}).astype('int32')
    data["capillary_refill_time"] = data["capillary_refill_time"].map({'less_3_sec': '0',
                                                                       'more_3_sec': '1'}).astype('int32')
    data["pain"] = data["pain"].map({'alert': '0', 'depressed': '1', 'moderate': '2', 'mild_pain': '3',
                                     'severe_pain': '4', 'extreme_pain': '5'}).astype('int32')
    data["peristalsis"] = data["peristalsis"].map({'hypermotile': '0', 'normal': '1',
                                                   'hypomotile': '2', 'absent': '3'}).astype('int32')
    data["abdominal_distention"] = data["abdominal_distention"].map({'none': '0', 'slight': '1', 'moderate': '2',
                                                                     'severe': '3'}).astype('int32')
    data["nasogastric_tube"] = data["nasogastric_tube"].map({'none': '0', 'slight': '1',
                                                             'significant': '2'}).astype('int32')
    data["nasogastric_reflux"] = data["nasogastric_reflux"].map({'none': '0',
                                                                 'less_1_liter': '1',
                                                                 'more_1_liter': '2'}).astype('int32')
    data["rectal_exam_feces"] = data["rectal_exam_feces"].map({'absent': '0', 'decreased': '1', 'normal': '2',
                                                               'increased': '3'}).astype('int32')
    data["abdomen"] = data["abdomen"].map({'normal': '0', 'other': '1', 'firm': '2', 'distend_small': '3',
                                           'distend_large': '4'}).astype('int32')
    data["abdomo_appearance"] = data["abdomo_appearance"].map({'clear': '0',
                                                               'cloudy': '1',
                                                               'serosanguious': '2'}).astype('int32')
    data["surgical_lesion"] = data["surgical_lesion"].map({'no': '0', 'yes': '1'}).astype('int32')

    #%% Numeric absent values imputation
    # cuML doesn't have any imputer for dask_cudf
    # dask_ml have only CPU based SimpleImputer
    # Using handmade imputer

    data[num_cols] = data[num_cols].astype('float64')

    if train:
        num_imputer = {}
        for col in num_cols:
            num_imputer[col] = data[col].dropna().mean().compute()
    else:
        num_imputer = load(os.path.join(RESOURCES_DIR, 'num_imputer_v0.1.joblib'))


    for col in num_cols:
        mask = data[col].isna()
        data[col] = data[col].mask(mask, num_imputer[col])

    del col, mask

    #%% OHE Columns absent values imputation
    # cuML doesn't have any imputer for dask_cudf
    # dask_ml have only CPU based SimpleImputer
    # Using handmade imputer

    data[ohe_cols] = data[ohe_cols].astype('float64')

    if train:
        ohe_imputer = {}
        for col in ohe_cols:
            ohe_imputer[col] = data[col].dropna().mean().compute()
    else:
        ohe_imputer = load(os.path.join(RESOURCES_DIR, 'ohe_imputer_v0.1.joblib'))

    for col in ohe_cols:
        mask = data[col].isna()
        data[col] = data[col].mask(mask, ohe_imputer[col])

    del col, mask

    # %%  One hot encoding of categorical attributes

    # This operation cannot be done on dask_cuda safely yet

    # Information on 01/11/23 in https://docs.rapids.ai/api/cuml/stable/api/#cuml.preprocessing.OneHotEncoder:
    # sparsebool, default=True - This feature is not fully supported by cupy yet,
    # causing incorrect values when computing one hot encodings. # See cupy/cupy#3223

    # The premise of the exercise of keeping all possible operation inside dask_cudf,
    # in order to assess its project complexities cannot be maintained at this point

    data = cudf.DataFrame(data.compute())

    if train:
        ohe_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ohe_encoder = ohe_encoder.fit(data[ohe_cols])
    else:
        ohe_encoder = load(os.path.join(RESOURCES_DIR, 'ohe_encoder_v0.1.joblib'))

    ohe_cols_data = ohe_encoder.transform(data[ohe_cols])
    ohe_cols_data = cudf.DataFrame(ohe_cols_data, columns=ohe_encoder.get_feature_names(ohe_cols),
                                   index=data[ohe_cols].index)
    ohe_cols_data = ohe_cols_data.set_index(data[ohe_cols].index)
    data = cudf.concat([data.drop(columns=ohe_cols), ohe_cols_data], axis=1)
    ohe_cols_encoded = ohe_encoder.get_feature_names(ohe_cols).tolist()
    data_cols = num_cols + ohe_cols_encoded

    del ohe_cols_data, ohe_cols, num_cols, ohe_cols_encoded


    # %% Drop constant features - Didn't have any effect, not implemented

    # %% Drop correlated features - cuML doesn't have a function for this
    # Using handmade filter

    if train:
        threshold = 0.75
        corr_matrix = data[data_cols].corr().abs()
        upper_tri_mask = cp.triu(cp.ones(corr_matrix.shape), k=1).astype(cp.bool_)
        upper_tri = corr_matrix.where(upper_tri_mask)
        drop_correlated_filter = upper_tri.columns[(upper_tri > threshold).any().values_host].tolist()
    else:
        drop_correlated_filter = load(os.path.join(RESOURCES_DIR, 'drop_correlated_filter_v0.1.joblib'))

    data.drop(columns=drop_correlated_filter, inplace=True)
    data_cols = [item for item in data_cols if item not in drop_correlated_filter]

    if train:
        del corr_matrix, upper_tri_mask, upper_tri, threshold


    #%% Left-skewed normalization

    left_skewed = ['abdomo_protein', 'pulse', 'respiratory_rate', 'total_protein']

    log1cp_transformer = FunctionTransformer(func=cp.log1p)
    data_normalized = log1cp_transformer.transform(data[left_skewed])
    data_normalized = data_normalized.reset_index(drop=True)
    data_normalized = data_normalized.set_index(data[left_skewed].index)
    data_normalized.columns = data[left_skewed].columns
    data = cudf.concat([data.drop(columns=left_skewed), data_normalized], axis=1)


    del log1cp_transformer, left_skewed, data_normalized

    #%% Standard scaling

    if train:
        standard_scaler_transformer = StandardScaler(copy=False, with_std=True)
        standard_scaler_transformer = standard_scaler_transformer.fit(data[data_cols])
    else:
        standard_scaler_transformer = load(os.path.join(RESOURCES_DIR, 'standard_scaler_transformer_v0.1.joblib'))

    data_scaled = standard_scaler_transformer.transform(data[data_cols])
    data_scaled = data_scaled.reset_index(drop=True)
    data_scaled = data_scaled.set_index(data[data_cols].index)
    data_scaled.columns = data[data_cols].columns
    data = cudf.concat([data.drop(columns=data_cols), data_scaled], axis=1)

    del data_scaled

    #%% Adjust numeric type for tuning models (Requisite of XGBoost)

    data = data.astype({c: 'float32' for c in data_cols})

    # %% Treat outcome

    if train:
        data['outcome'] = data['outcome'].replace({'died': '2', 'euthanized': '1', 'lived': '0'}).astype('int32')
        outcome = data.pop('outcome')
        data['outcome'] = outcome


    #%% Save resources
    if train:
        dump(hospital_number_codifier, os.path.join(RESOURCES_DIR, 'hospital_number_codifier_v0.1.joblib'))
        dump(num_imputer, os.path.join(RESOURCES_DIR, 'num_imputer_v0.1.joblib'))
        dump(ohe_imputer, os.path.join(RESOURCES_DIR, 'ohe_imputer_v0.1.joblib'))
        dump(ohe_encoder, os.path.join(RESOURCES_DIR, 'ohe_encoder_v0.1.joblib'))
        dump(drop_correlated_filter, os.path.join(RESOURCES_DIR, 'drop_correlated_filter_v0.1.joblib'))
        dump(standard_scaler_transformer, os.path.join(RESOURCES_DIR, 'standard_scaler_transformer_v0.1.joblib'))


    #%%

    return data

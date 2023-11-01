import sys

sys.path.append("../")

import os
import cudf
import cupy as cp
from joblib import dump, load

from B_resources.c_CUDA_RAPIDS.lesion_encoder import (lesion_site, lesion_code, lesion_type, lesion_sub_type,
                                                      lesion_number)

from cuml.preprocessing import OneHotEncoder, SimpleImputer, FunctionTransformer, StandardScaler


# %% Clean data

def scrub_feature_engineering(data, train=False):
    #system setup
    BASE_DIR = os.path.dirname(os.getcwd())
    RESOURCES_DIR = os.path.join(BASE_DIR, 'B_resources', 'c_CUDA_RAPIDS')

    # Drop cp_data, without information
    data.drop(columns=['cp_data'], inplace=True)

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

    # %% Function - Lesion Treatment
    data['lesionSite'] = data['lesion_1'].apply(lesion_site)
    data['lesionType'] = data['lesion_1'].apply(lesion_type)
    data['lesionSubType'] = data['lesion_1'].apply(lesion_sub_type)
    data['lesionCode'] = data['lesion_1'].apply(lesion_code)
    data['numLesions'] = data.apply(lesion_number)

    data.drop(columns=['lesion_1', 'lesion_2', 'lesion_3'], inplace=True)
    ohe_cols = [item for item in ohe_cols if item not in ['lesion_1', 'lesion_2', 'lesion_3']]
    ohe_cols = ohe_cols + ['lesionSite', 'lesionType', 'lesionSubType', 'lesionCode']
    num_cols = num_cols + ['numLesions']

    # %% Coding hospital numbers

    if train:
        hospital_number_codifier = cudf.DataFrame(data['hospital_number'].value_counts().reset_index())
        hospital_number_codifier.columns = ["hospital_number", "freq"]
        hospital_number_codifier = hospital_number_codifier.sort_values(by="freq", ascending=False)
    else:
        hospital_number_codifier = load(os.path.join(RESOURCES_DIR, 'hospital_number_codifier_v0.1.joblib'))
        mask = ~data['hospital_number'].isin(hospital_number_codifier['hospital_number'])
        data.loc[mask, 'hospital_number'] = hospital_number_codifier['hospital_number'].iloc[0]
        del mask

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

    # %% Impute absent values in numeric columns
    # cuML doesn't provide a KNNImputer, using SimpleImputer instead

    if train:
        num_imputer = SimpleImputer(strategy='median', copy=False)
        num_imputer.fit(data[num_cols])
    else:
        num_imputer = load(os.path.join(RESOURCES_DIR, 'num_imputer_v0.1.joblib'))

    data[num_cols] = num_imputer.transform(data[num_cols])

    if train:
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

    # %% Impute absent values in ohe columns

    if train:
        ohe_imputer = SimpleImputer(strategy='median', copy=False)
        ohe_imputer.fit(data[ohe_cols])
    else:
        ohe_imputer = load(os.path.join(RESOURCES_DIR, 'ohe_imputer_v0.1.joblib'))

    data[ohe_cols] = ohe_imputer.transform(data[ohe_cols])

    if train:
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

    # %%  One hot encoding of categorical attributes

    if train:
        ohe_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ohe_encoder = ohe_encoder.fit(data[ohe_cols])
    else:
        ohe_encoder = load(os.path.join(RESOURCES_DIR, 'ohe_encoder_v0.1.joblib'))

    ohe_cols_data = ohe_encoder.transform(data[ohe_cols])
    ohe_cols_data = cudf.DataFrame(ohe_cols_data, columns=ohe_encoder.get_feature_names(ohe_cols))
    data = cudf.concat([data.drop(columns=ohe_cols), ohe_cols_data], axis=1)
    ohe_cols_encoded = ohe_encoder.get_feature_names(ohe_cols).tolist()
    data_cols = num_cols + ohe_cols_encoded

    del ohe_cols_data, ohe_cols, num_cols, ohe_cols_encoded

    # %% Drop constant features - Didn't have any effect, not implemented

    # %% Drop correlated features - cuML doesn't have a function for this

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
    data[left_skewed] = log1cp_transformer.transform(data[left_skewed])
    data[left_skewed].columns = left_skewed

    del log1cp_transformer, left_skewed

    #%% Standard scaling

    if train:
        standard_scaler_transformer = StandardScaler(copy=False, with_std=True)
        standard_scaler_transformer = standard_scaler_transformer.fit(data[data_cols])
    else:
        standard_scaler_transformer = load(os.path.join(RESOURCES_DIR, 'standard_scaler_transformer_v0.1.joblib'))

    data[data_cols] = standard_scaler_transformer.transform(data[data_cols])
    data[data_cols].columns = data_cols

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


    return data
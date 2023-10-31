import sys

sys.path.append("../")

import pandas as pd
from joblib import dump, load

from B_resources.b_Python_sklearn.lesion_encoder import lesion_encode

from sklearnex import patch_sklearn  # Intel(R) Extension for Scikit-learn

patch_sklearn()  # Intel(R) Extension for Scikit-learn

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
from feature_engine.transformation import LogCpTransformer


#%%

def preparation(data, train = False):
    data.drop(columns=['cp_data'], inplace=True)
    data[['hospital_number', 'lesion_1', 'lesion_2', 'lesion_3']] = (
        data[['hospital_number', 'lesion_1', 'lesion_2', 'lesion_3']].astype(str))
    ohe_cols = data.select_dtypes(include=['object']).columns.to_list()
    if train:
        ohe_cols = [item for item in ohe_cols if item not in ['outcome']]
    num_cols = data.select_dtypes(include=['number']).columns.to_list()

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

    # %% Codify lesions

    data, ohe_cols, num_cols = lesion_treatment(data, ohe_cols, num_cols)
    ohe_cols = ohe_cols + ['lesionSite', 'lesionType', 'lesionSubType', 'lesionCode']

    # %% Coding hospital numbers

    if train:
        hospital_number_codifier = pd.DataFrame(data['hospital_number'].value_counts().reset_index())
        hospital_number_codifier.columns = ["hospital_number", "freq"]
        hospital_number_codifier = hospital_number_codifier.sort_values(by="freq", ascending=False)
    else:
        hospital_number_codifier = load('../resources/hospital_number_codifier_v0.1.joblib')
        mask = ~data['hospital_number'].isin(hospital_number_codifier['hospital_number'])
        data.loc[mask, 'hospital_number'] = hospital_number_codifier['hospital_number'].iloc[0]

    # %% Impute absent values in numeric columns

    data_numeric = data[num_cols].copy()
    if train:
        num_imputer = KNNImputer(n_neighbors=10)
        num_imputer.fit(data_numeric)
    else:
        num_imputer = load('../resources/num_imputer_v0.1.joblib')

    data_numeric_imputed = num_imputer.transform(data_numeric)
    data_numeric_imputed_df = pd.DataFrame(data_numeric_imputed, columns=num_cols, index=data_numeric.index)
    data_numeric_imputed_df = data_numeric_imputed_df.astype(data_numeric.dtypes)

    data = pd.concat([data.drop(columns=num_cols), data_numeric_imputed_df], axis=1)

    if train:
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

    del data_numeric, data_numeric_imputed, data_numeric_imputed_df

    #%% Impute categorical attributes

    data_ohe = data[ohe_cols].copy()

    if train:
        ohe_imputer = SimpleImputer(strategy='most_frequent')
        ohe_imputer = ohe_imputer.fit(data_ohe)
    else:
        ohe_imputer = load('../resources/ohe_imputer_v0.1.joblib')

    data_imputed = ohe_imputer.transform(data_ohe)
    data_imputed_pd = pd.DataFrame(data_imputed, columns=ohe_cols, index=data_ohe.index)
    data = pd.concat([data.drop(columns=ohe_cols), data_imputed_pd], axis=1)

    if train:
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

    del data_ohe, data_imputed, data_imputed_pd

    #%% One hot encoding of categorical attributes
    data_ohe = data[ohe_cols].copy()

    if train:
        ohe_encoder = OneHotEncoder(sparse_output=False)
        ohe_encoder = ohe_encoder.fit(data_ohe)
    else:
        ohe_encoder = load('../resources/ohe_encoder_v0.1.joblib')

    data_encoded = ohe_encoder.transform(data_ohe)
    data_encoded_pd = pd.DataFrame(data_encoded, columns=ohe_encoder.get_feature_names_out(ohe_cols),
                                   index=data_ohe.index)
    data = pd.concat([data.drop(columns=ohe_cols), data_encoded_pd], axis=1)

    ohe_cols_encoded = ohe_encoder.get_feature_names_out(ohe_cols).tolist()
    data_cols = num_cols + ohe_cols_encoded
    del data_ohe, data_encoded, data_encoded_pd, num_cols, ohe_cols, ohe_cols_encoded

    #%% Drop constant features
    data_const = data[data_cols].copy()

    if train:
        drop_zv_transformer = DropConstantFeatures()
        drop_zv_transformer = drop_zv_transformer.fit(data_const)
    else:
        drop_zv_transformer = load('../resources/drop_zv_transformer_v0.1.joblib')

    data_const_dropped = drop_zv_transformer.transform(data_const)
    data_zv_cols = data_const_dropped.columns.to_list()
    data = pd.concat([data.drop(columns=data_cols), data_const_dropped], axis=1)
    del data_const, data_const_dropped, data_cols

    #%% Drop correlated features
    data_corr = data[data_zv_cols].copy()

    if train:
        drop_corr_transformer = DropCorrelatedFeatures()
        drop_corr_transformer = drop_corr_transformer.fit(data_corr)
    else:
        drop_corr_transformer = load('../resources/drop_corr_transformer_v0.1.joblib')

    data_corr_dropped = drop_corr_transformer.transform(data_corr)
    data = pd.concat([data.drop(columns=data_zv_cols), data_corr_dropped], axis=1)
    data_corr_cols = data_corr_dropped.columns.to_list()
    del data_corr, data_corr_dropped, data_zv_cols

    #%% Left-skewed normalization
    left_skewed = ['abdomo_protein', 'pulse', 'respiratory_rate', 'total_protein']
    data_left_skewed = data[left_skewed].copy()

    if train:
        left_skewed_transformer = LogCpTransformer()
        left_skewed_transformer = left_skewed_transformer.fit(data_left_skewed)
    else:
        left_skewed_transformer = load('../resources/left_skewed_transformer_v0.1.joblib')

    data_left_skewed_norm = left_skewed_transformer.transform(data_left_skewed)
    data = pd.concat([data.drop(columns=left_skewed), data_left_skewed_norm], axis=1)
    del data_left_skewed, data_left_skewed_norm, left_skewed

    #%% Standard scaling
    data_scale = data[data_corr_cols].copy()

    if train:
        scaler_transformer = StandardScaler()
        scaler_transformer = scaler_transformer.fit(data_scale)
    else:
        scaler_transformer = load('../resources/scaler_transformer_v0.1.joblib')

    data_scale_transformed = scaler_transformer.transform(data_scale)
    data_scale_transformed_pd = pd.DataFrame(data_scale_transformed,
                                             columns=scaler_transformer.get_feature_names_out(data_corr_cols),
                                             index=data_scale.index)
    data = pd.concat([data.drop(columns=data_corr_cols), data_scale_transformed_pd], axis=1)

    data_scaled_cols = scaler_transformer.get_feature_names_out(data_corr_cols).tolist()
    del data_scale, data_scale_transformed, data_scale_transformed_pd, data_corr_cols


    # %% Treat outcome

    if train:
        data['outcome'] = data['outcome'].replace({'died': 2, 'euthanized': 1, 'lived': 0})
        data['outcome'] = pd.Categorical(data['outcome'], categories=[2, 1, 0], ordered=True)
        outcome = data.pop('outcome')
        data['outcome'] = outcome

    #%% Save resources
    if train:
        dump(hospital_number_codifier, '../resources/hospital_number_codifier_v0.1.joblib')
        dump(num_imputer, '../resources/num_imputer_v0.1.joblib')
        dump(ohe_imputer, '../resources/ohe_imputer_v0.1.joblib')
        dump(ohe_encoder, '../resources/ohe_encoder_v0.1.joblib')
        dump(drop_zv_transformer, '../resources/drop_zv_transformer_v0.1.joblib')
        dump(drop_corr_transformer, '../resources/drop_corr_transformer_v0.1.joblib')
        dump(left_skewed_transformer, '../resources/left_skewed_transformer_v0.1.joblib')
        dump(scaler_transformer, '../resources/scaler_transformer_v0.1.joblib')

    return data

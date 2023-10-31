import pandas as pd


def lesion_encode(df, ohe_cols, num_cols):
    df['numLesions'] = (df[['lesion_1', 'lesion_2', 'lesion_3']] != "0").sum(axis=1)
    num_cols = num_cols + ['numLesions']

    df['lesionSite'] = -1
    df['lesionType'] = -1
    df['lesionSubType'] = -1
    df['lesionCode'] = -1

    for i, row in df.iterrows():
        for lesion_col in ['lesion_1', 'lesion_2', 'lesion_3']:
            lesion_val = row[lesion_col]
            if lesion_val != "" and lesion_val != 'nan':
                if len(lesion_val) == 5:
                    if lesion_val[:2] == "11":
                        df.at[i, 'lesionSite'] = 11
                        df.at[i, 'lesionType'] = int(lesion_val[2])
                        df.at[i, 'lesionSubType'] = int(lesion_val[3])
                        df.at[i, 'lesionCode'] = int(lesion_val[4])
                    else:
                        if df.at[i, 'lesionSite'] < int(lesion_val[0]):
                            df.at[i, 'lesionSite'] = int(lesion_val[0])
                            df.at[i, 'lesionType'] = int(lesion_val[1])
                            df.at[i, 'lesionSubType'] = int(lesion_val[2])
                            df.at[i, 'lesionCode'] = int(lesion_val[3:5])
                else:
                    if df.at[i, 'lesionSite'] < int(lesion_val[0]):
                        lesion_val = lesion_val.ljust(4, '0')
                        df.at[i, 'lesionSite'] = int(lesion_val[0])
                        df.at[i, 'lesionType'] = int(lesion_val[1])
                        df.at[i, 'lesionSubType'] = int(lesion_val[2])
                        df.at[i, 'lesionCode'] = int(lesion_val[3])

    df.drop(['lesion_1', 'lesion_2', 'lesion_3'], axis=1, inplace=True)
    ohe_cols = [item for item in ohe_cols if item not in ['lesion_1', 'lesion_2', 'lesion_3']]

    lesionSiteDict = {"11": "all intestinal sites",
                      "1": "gastric",
                      "2": "sm intestine",
                      "3": "lg colon",
                      "4": "lg colon and cecum",
                      "5": "cecum",
                      "6": "transverse colon",
                      "7": "retum/descending colon",
                      "8": "uterus",
                      "9": "bladder",
                      "0": "none",
                      "-1": "n/a"}

    lesionTypeDict = {"1": "simple",
                      "2": "strangulation",
                      "3": "inflammation",
                      "4": "other",
                      "5": "other",
                      "6": "other",
                      "7": "other",
                      "8": "other",
                      "9": "other",
                      "0": "other",
                      "-1": "n/a"}

    lesionSubTypeDict = {"1": "mechanical",
                         "2": "paralytic",
                         "3": "n/a",
                         "4": "n/a",
                         "5": "n/a",
                         "6": "n/a",
                         "7": "n/a",
                         "8": "n/a",
                         "9": "n/a",
                         "0": "n/a",
                         "-1": "n/a"}

    lesionCodeDict = {"10": "displacement",
                      "1": "obturation",
                      "2": "intrinsic",
                      "3": "extrinsic",
                      "4": "adynamic",
                      "5": "volvulus/torsion",
                      "6": "intussuption",
                      "7": "thromboembolic",
                      "8": "hernia",
                      "9": "lipoma/slenic incarceration",
                      "0": "n/a",
                      "-1": "n/a"}

    df['lesionSite'] = df['lesionSite'].astype(str)
    df['lesionType'] = df['lesionType'].astype(str)
    df['lesionSubType'] = df['lesionSubType'].astype(str)
    df['lesionCode'] = df['lesionCode'].astype(str)

    df['lesionSite'] = df['lesionSite'].replace(lesionSiteDict)
    df['lesionType'] = df['lesionType'].replace(lesionTypeDict)
    df['lesionSubType'] = df['lesionSubType'].replace(lesionSubTypeDict)
    df['lesionCode'] = df['lesionCode'].replace(lesionCodeDict)

    return df, ohe_cols, num_cols

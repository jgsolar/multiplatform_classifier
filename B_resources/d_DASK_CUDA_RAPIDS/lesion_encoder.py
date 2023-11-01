import cudf
import cupy as cp


def code_lesion_site(lesion: cudf.Series) -> cudf.Series:
    site = cudf.Series(cp.zeros(len(lesion), dtype=int))
    site.index = lesion.index

    lesion_12000 = (lesion >= 12000)
    site[lesion_12000] = ((lesion[lesion_12000] // 10) // (10 ** 3)) % 10
    lesion_12000_11000_site_11 = (12000 > lesion) & (lesion >= 11000)
    site[lesion_12000_11000_site_11] = 11
    lesion_11000_10000_code_10 = (11000 > lesion) & (lesion >= 10000)
    site[lesion_11000_10000_code_10] = ((lesion[lesion_11000_10000_code_10] // 10) // (10 ** 3)) % 10
    lesion_10000_1000 = (10000 > lesion) & (lesion >= 1000)
    site[lesion_10000_1000] = (lesion[lesion_10000_1000] // (10 ** 3)) % 10
    lost_1digit = (1000 > lesion) & (lesion >= 100)  # Anomaly lesion
    site[lost_1digit] = ((lesion[lost_1digit] * 10) // (10 ** 3)) % 10
    lost_2digit = (100 > lesion) & (lesion >= 10)  # Anomaly lesion
    site[lost_2digit] = ((lesion[lost_2digit] * 100) // (10 ** 3)) % 10
    lost_3digit = (10 > lesion) & (lesion >= 1)  # Anomaly lesion
    site[lost_3digit] = ((lesion[lost_3digit] * 1000) // (10 ** 3)) % 10

    return site


def code_lesion_type(lesion: cudf.Series) -> cudf.Series:
    les_type = cudf.Series(cp.zeros(len(lesion), dtype=int))
    les_type.index = lesion.index

    lesion_12000_code_10 = (lesion >= 12000)
    les_type[lesion_12000_code_10] = ((lesion[lesion_12000_code_10] // 10) // (10 ** 2)) % 10
    lesion_12000_11000_site_11 = (12000 > lesion) & (lesion >= 11000)
    les_type[lesion_12000_11000_site_11] = (lesion[lesion_12000_11000_site_11] // (10 ** 2)) % 10
    lesion_11000_10000_code_10 = (11000 > lesion) & (lesion >= 10000)
    les_type[lesion_11000_10000_code_10] = ((lesion[lesion_11000_10000_code_10] // 10) // (10 ** 2)) % 10
    lesion_10000_1000 = (10000 > lesion) & (lesion >= 1000)
    les_type[lesion_10000_1000] = (lesion[lesion_10000_1000] // (10 ** 2)) % 10
    lost_1digit = (1000 > lesion) & (lesion >= 100)  # Anomaly lesion
    les_type[lost_1digit] = ((lesion[lost_1digit] * 10) // (10 ** 2)) % 10
    lost_2digit = (100 > lesion) & (lesion >= 10)  # Anomaly lesion
    les_type[lost_2digit] = ((lesion[lost_2digit] * 100) // (10 ** 2)) % 10
    lost_3digit = (10 > lesion) & (lesion >= 1)  # Anomaly lesion
    les_type[lost_3digit] = ((lesion[lost_3digit] * 1000) // (10 ** 2)) % 10

    return les_type


def code_lesion_sub_type(lesion: cudf.Series) -> cudf.Series:
    subtype = cudf.Series(cp.zeros(len(lesion), dtype=int))
    subtype.index = lesion.index

    lesion_12000_code_10 = (lesion >= 12000)
    subtype[lesion_12000_code_10] = ((lesion[lesion_12000_code_10] // 10) // (10 ** 1)) % 10
    lesion_12000_11000_site_11 = (12000 > lesion) & (lesion >= 11000)
    subtype[lesion_12000_11000_site_11] = (lesion[lesion_12000_11000_site_11] // (10 ** 1)) % 10
    lesion_11000_10000_code_10 = (11000 > lesion) & (lesion >= 10000)
    subtype[lesion_11000_10000_code_10] = ((lesion[lesion_11000_10000_code_10] // 10) // (10 ** 1)) % 10
    lesion_10000_1000 = (10000 > lesion) & (lesion >= 1000)
    subtype[lesion_10000_1000] = (lesion[lesion_10000_1000] // (10 ** 1)) % 10
    lost_1digit = (1000 > lesion) & (lesion >= 100)  # Anomaly lesion
    subtype[lost_1digit] = ((lesion[lost_1digit] * 10) // (10 ** 1)) % 10
    lost_2digit = (100 > lesion) & (lesion >= 10)  # Anomaly lesion
    subtype[lost_2digit] = ((lesion[lost_2digit] * 100) // (10 ** 1)) % 10
    lost_3digit = (10 > lesion) & (lesion >= 1)  # Anomaly lesion
    subtype[lost_3digit] = ((lesion[lost_3digit] * 1000) // (10 ** 1)) % 10

    return subtype


def code_lesion_code(lesion: cudf.Series) -> cudf.Series:
    code = cudf.Series(cp.zeros(len(lesion), dtype=int))
    code.index = lesion.index

    lesion_12000_code_10 = (lesion >= 12000)
    code[lesion_12000_code_10] = 10
    lesion_12000_11000_site_11 = (12000 > lesion) & (lesion >= 11000)
    code[lesion_12000_11000_site_11] = (lesion[lesion_12000_11000_site_11] // (10 ** 0)) % 10
    lesion_11000_10000_code_10 = (11000 > lesion) & (lesion >= 10000)
    code[lesion_11000_10000_code_10] = 10
    lesion_10000_1000 = (10000 > lesion) & (lesion >= 1000)
    code[lesion_10000_1000] = (lesion[lesion_10000_1000] // (10 ** 0)) % 10
    lost_1digit = (1000 > lesion) & (lesion >= 100)  # Anomaly lesion
    code[lost_1digit] = ((lesion[lost_1digit] * 10) // (10 ** 0)) % 10
    lost_2digit = (100 > lesion) & (lesion >= 10)  # Anomaly lesion
    code[lost_2digit] = ((lesion[lost_2digit] * 100) // (10 ** 0)) % 10
    lost_3digit = (10 > lesion) & (lesion >= 1)  # Anomaly lesion
    code[lost_3digit] = ((lesion[lost_3digit] * 1000) // (10 ** 0)) % 10

    return code


def lesion_number(lesions: cudf.DataFrame) -> int:
    lesion_1 = lesions['lesion_1']
    lesion_2 = lesions['lesion_2']
    lesion_3 = lesions['lesion_3']

    num_lesion = cudf.Series(cp.zeros(len(lesion_1), dtype=int))
    num_lesion.index = lesion_1.index

    single_lesion = (lesion_1 != 0) & (lesion_2 == 0) & (lesion_3 == 0)
    num_lesion[single_lesion] = 1
    double_lesion = (lesion_1 != 0) & (lesion_2 != 0) & (lesion_3 == 0)
    num_lesion[double_lesion] = 2
    triple_lesion = (lesion_1 != 0) & (lesion_2 != 0) & (lesion_3 != 0)
    num_lesion[triple_lesion] = 3

    return num_lesion




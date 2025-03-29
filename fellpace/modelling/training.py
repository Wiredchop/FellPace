import numpy as np


def get_race_coeffs(data_Zs, use_inliers=True):
    if use_inliers:
        data_Zs = data_Zs.loc[data_Zs['inlier'] == True]

    return data_Zs.groupby('Race_Name').apply(lambda x: np.polyfit(x['ZScore'], x['HCScore'], 1))


import pandas as pd
from sklearn import linear_model


def find_inliers(group):
    X = group.ZScore.values.reshape([-1,1])
    y = group.HCScore.values.reshape([-1,1])
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    return pd.Series(ransac.inlier_mask_)


def add_inliers(data_Zs):
    data_Zs['inlier'] = data_Zs.groupby('Race_Name', group_keys=False, sort=True, as_index=False).apply(find_inliers).values
    return data_Zs
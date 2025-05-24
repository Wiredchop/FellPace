import pandas as pd
from sklearn import linear_model

# Some of the more relaxed races find poor models
# If the race name is in the list below, limit the fit of the model to 1st coeff > 1
# This is inline with the more relaxed races
modify_races = ['PR_Endcliffe']

def force_over_1(fit, *args):
    return fit.coef_[0][0] > 1

def find_inliers(group):
    X = group.ZScore.values.reshape([-1,1])
    y = group.HCScore.values.reshape([-1,1])
    if group.name in modify_races:
        ransac = linear_model.RANSACRegressor(is_model_valid=force_over_1, max_trials=1000)
    else:
        ransac = linear_model.RANSACRegressor()
        
    ransac.fit(X, y)
    return pd.Series(ransac.inlier_mask_)


def add_inliers(data_Zs):
    data_Zs['inlier'] = data_Zs.groupby('Race_Name', group_keys=False, sort=True, as_index=False).apply(find_inliers).values
    return data_Zs


import matplotlib.pyplot as plt
import seaborn as sns


def plot_inlier_outlier(x, y, inlier, **kwargs):
    x_in = x.loc[inlier == True]
    y_in = y.loc[inlier == True]
    x_out = x.loc[inlier == False]
    y_out = y.loc[inlier == False]
    sns.regplot(x=x_in, y=y_in)
    sns.scatterplot(x=x_out, y=y_out, s=60, color=[0.6, 0.6, 0.6], alpha=0.5).set(xlabel='Race performance', ylabel='Chase performance')


def plot_all_race_Zscores(data_Zs):
    g = sns.FacetGrid(data=data_Zs, col='Race_Name', col_wrap=4).set_titles("{col_name}")
    g.map(plot_inlier_outlier, "ZScore", "HCScore", "inlier")
    g.set_xlabels('Race Performance').set_ylabels('Chase Performance')
    plt.show()
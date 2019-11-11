import pandas as pd
from numpy import ones, array
from scipy.stats import f as stats_f, shapiro, ttest_rel
from itertools import combinations
from .sphericity_functions import Greenhouse_Geisser, Huynh_Feldt

_shapiro = shapiro  # backup


def shapiro(x):
    """
    Wrapper for the scipy.stats.shapiro function to allow for use the pandas.DataFrame.apply function
    """
    W, p = _shapiro(x)
    return pd.Series([W, p], index=['W', 'p'])


def anova(df, corr='GG', print_table=True, p_normal=0.05):
    """
    Repeated measures Analysis of Variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with conditions or times as columns, and subjects as rows
    corr : str, optional
        Which correction to apply for sphericity.  Options are 'GG' (default) for Greenhouse-Geisler,
        'HF' for Huynh-Feldt, 'avg' for the average of 'GG' and 'HF', or None for no correction.
    print_table : bool, optional
        Print a descriptive table with the statistics and normality test (Shapiro-Wilk) results.
    p_normal : float, None, optional
        Normality test p-value from the Shapiro-Wilk normality test.  Defaults to 0.05, can be set to None to not
        test for normality.

    Returns
    -------
    F : float
        Test statistic
    p : float
        p-value for the test.
    eta : float
        Effect size.
    """
    pd.options.display.max_columns = 10
    pd.options.display.width = 1200
    stat = pd.DataFrame(index=['None', 'Greenhouse-Geissler', 'Huynh-Feldt', 'Average'])  # data frame for statistics

    n, k = df.shape  # number of subjects, number of conditions
    tot_mean = df.mean().mean()  # mean of all values

    # Variations
    ss_cond = n*((df.mean()-tot_mean)**2).sum()  # condition variation
    ss_w = ((df - df.mean())**2).sum().sum()  # within groups variation
    ss_subj = k*((df.mean(axis=1)-tot_mean)**2).sum()  # within subjects variation
    ss_error = ss_w - ss_subj  # error variation

    # Sphericity Corrections
    stat['epsilon'] = ones(4)  # list of sphericity estimates
    stat['epsilon'][1] = Greenhouse_Geisser(df)  # calculate Greenhouse-Geissler estimate of sphericity
    stat['epsilon'][2] = Huynh_Feldt(df)  # calculate Huynh-Feldt estimate of sphercity
    stat['epsilon'][3] = (stat['epsilon'][1] + stat['epsilon'][2])/2  # calculate average of GG and HF estimates

    # Degrees of Freedom
    df_cond = k - 1  # Condition degrees of freedom
    df_error = (n-1) * (k-1)  # Error degrees of freedom

    stat['Cond_DoF'] = stat['epsilon']*df_cond
    stat['Error_DoF'] = stat['epsilon']*df_error

    # Mean Sum of Squares
    ms_cond = ss_cond / df_cond  # mean sum of squares for conditions
    ms_error = ss_error / df_error  # mean sum of squares for error

    # Test Statistic and p-value.  Due to the math, correction is canceled in the F statistic
    stat['F'] = ones(4) * (ms_cond / ms_error)

    stat['p-value'] = stats_f.sf(stat['F'], stat['Cond_DoF'], stat['Error_DoF'])

    # Effect Size
    stat['eta'] = ones(4) * (ss_cond/(ss_cond + ss_error))

    stat = stat[['epsilon', 'Cond_DoF', 'Error_DoF', 'eta', 'F', 'p-value']]  # re-arrange for easier viewing

    # Normality Tests
    if p_normal is not None:
        normality = pd.DataFrame(df.apply(shapiro, axis=0)).transpose()
        normality['Normal'] = normality['p'] > p_normal

    # Display statistics table
    if print_table:
        print(stat)
        print('\n')
        print(normality)

    if corr == 'GG':
        return stat['F'][0], stat['p-value'][1], stat['eta'][0]
    elif corr == 'HF':
        return stat['F'][0], stat['p-value'][2], stat['eta'][0]
    elif corr == 'avg':
        return stat['F'][0], stat['p-value'][3], stat['eta'][0]
    else:
        return stat['F'][0], stat['p-value'][0], stat['eta'][0]


def combinations_t_test(df, alpha=0.05, one_sided=True):
    """
    Individual dependent t-tests of all possible combinations of different timestamps.  Uses scipy.stats.ttest_rel.
    This performs a two-sided t-test, however the one-sided result can be obtained from the sign of the test
    statistic and half of the returned p-value.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of data to analyze.  Each column corresponds to a different timestamp/condition of data.
    alpha : float, optional
        alpha level for determining if the null hypothesis can be rejected.  Defaults to 0.05.
    one_sided : bool, optional
        Return one-sided or two-sided interpretation of results in the output.  Defaults to True (return one-sided).

    Returns
    -------
    comb_stats : pandas.DataFrame
        DataFrame containing the test statistic and p-value for all the combinations of time stamps/conditions
        present in the DataFrame columns.
    """
    cols = df.columns  # time stamps/conditions to test between
    combs = list(combinations(cols, 2))  # get all the length 2 combinations of the columns of the dataFrame

    # create a DataFrame to store the test results
    ci = pd.MultiIndex.from_tuples(combs, names=['Cond. 1', 'Cond. 2'])  # create a heirarchical index
    comb_stats = pd.DataFrame(index=ci, columns=['T', 'p'])

    for cond1, cond2 in combs:
        comb_stats.loc[cond1, cond2] = ttest_rel(df[cond1], df[cond2])

    comb_stats['Comparison'] = "="  # add empty column for direction of comparison result
    if one_sided:
        # if test statistic is positive, cond 1 > cond 2
        comb_stats.loc[(comb_stats.p.values <= alpha) & (comb_stats['T'].values > 0), 'Comparison'] = '>'
        # if test statistic is negative, cond 1 < cond 2
        comb_stats.loc[(comb_stats.p.values <= alpha) & (comb_stats['T'].values < 0), 'Comparison'] = '<'
    else:
        # if the p-value < alpha, the distributions do not have equivalent expected values
        comb_stats.loc[(comb_stats.p.values <= alpha), 'Comparison'] = '!='

    return comb_stats



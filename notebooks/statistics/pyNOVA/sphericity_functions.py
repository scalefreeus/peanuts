from numpy import diag, sum


def Greenhouse_Geisser(df):
    """
    Greenhouse-Geisser estimate of the sphericity.  Implemented from the method in
    http://www.utd.edu/~herve/abdi-GreenhouseGeisser2010-pretty.pdf

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe object with conditions as columns and subjects as rows

    Returns
    -------
    eps : float
        Estimate of the sphericity
    """
    cov = df.cov()  # covariance matrix
    cov_mean = cov.mean()  # mean of covariance columns
    tot_mean = cov_mean.mean()  # total mean of covariance matrix

    S = cov.subtract(cov_mean, axis=0) - cov_mean + tot_mean  # double centered covariance matrix

    A, _ = S.shape  # number of indices
    eps = diag(S).sum()**2/((A-1)*sum(S**2).sum())
    return eps


def Huynh_Feldt(df):
    """
    Implementation of the Huynh-Feldt estimate of spheriticy.  Implemented from the method in
    http://www.utd.edu/~herve/abdi-GreenhouseGeisser2010-pretty.pdf

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe object with conditions as columns and subjects as rows

    Returns
    -------
    eps : float
        Estimate of the sphericity
    """
    S, A = df.shape  # shape of the dataframe

    _eps = Greenhouse_Geisser(df)  # Greenhouse-Geisser estimate of epsilon

    eps = (S*(A-1)*_eps - 2)/((A-1)*(S-1-(A-1)*_eps))

    return eps

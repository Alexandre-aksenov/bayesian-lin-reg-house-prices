"""
Functions for fitting and predicting from a bayesian linear model.

The signature of these functions is analogous to sklearn.

Wrapping these functions into a class
(making the posterior distributions of coefficients 'idata'
a class property) can make
interaction with these functions more user-friendly.
This improvement of style is a possible continnuation and can be  done
if the current model proves useful enough.


For a test of the functions: see the script train_test_split.ipynb .
"""

import pymc as pm
from functools import partial
# for producing the column names in posterior predictive distributions.


def fit(norm_train_features, observed, random_seed=42):
    """
    Fit the bayesian model.

    Draws samples from the posterior distributions of coefficients
    of a bayesian linear model. This function presupposes that the features
    and the target have been normalized.

    Args:
        norm_train_features (DataFrame): the features used to train the model.
        observed (DataFrame with 1 column): the target values.
        random_seed (int, optional): the random seed. Defaults to 42.

    Returns:
        InferenceData: the posterior distributions of the model parameters:
            Intercept, coefficients (beta[k,0]), sigma
    """
    with pm.Model():

        # Define priors
        sigma = pm.HalfNormal("sigma", sigma=1)
        intercept = pm.Normal("Intercept", 0, sigma=0.1)
        # relatively strong prior for the target variable has been normalized.

        beta = pm.Normal(
            "beta", 0, sigma=1, shape=(norm_train_features.shape[1], 1))

        pm.Normal(
            "y", mu=intercept + pm.math.dot(norm_train_features, beta),
            sigma=sigma, observed=observed)

        # Inference!
        # draw 1000 posterior samples using NUTS sampling
        idata = pm.sample(random_seed=random_seed)
    return idata


def predict(idata, norm_train_features, norm_test_features, random_seed=1):
    """
    The prediction is adapted from:
    # https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/

    Args:
        idata (InferenceData):
            the posterior distribution of parameters of the model.
        norm_train_features (DataFrame): the features used to train the model.
        norm_test_features (DataFrame): the new data for applying the model.
        random_seed (int, optional): the random seed. Defaults to 1.

    Returns:
        InferenceData: the posterior predictive distributions
            for both train and test points.
    """
    with pm.Model():
        # Define the same priors
        sigma = pm.HalfNormal("sigma", sigma=1)
        intercept = pm.Normal("Intercept", 0, sigma=0.1)
        beta = pm.Normal(
            "beta", 0, sigma=1, shape=(norm_train_features.shape[1], 1))

        # put all features together, the style should be improved!
        all_features = pm.math.concatenate(
            (norm_train_features, norm_test_features))

        pm.Normal(
            "y",
            mu=intercept + pm.math.dot(all_features, beta),
            sigma=sigma)

        # Sample using the previous 'idata'
        pp = pm.sample_posterior_predictive(
            idata, var_names=["y"], random_seed=random_seed)
    return pp


def ind_to_col_name(feat_index: int, posterior=True):
    """
    Returns a column name of df_idata_full in the format:
    ('posterior', 'beta[feat_index,0]', feat_index, 0)
    if 'posterior'
    OR (else)
    ('y[feat_index,0]', feat_index, 0).

    This function is used for extracting the posterior distributions
    of coefficients (posterior=True)
    OR
    of the posterior predictive distributions of the normalized log-price.
    """

    distribution_name = 'beta' if posterior else 'y'
    lst_distribution_type = ['posterior'] if posterior else []
    str_distribution_name = distribution_name + '[' + str(feat_index) + ',0]'

    lst_col_name = (lst_distribution_type
                    + [str_distribution_name, feat_index, 0])
    return tuple(lst_col_name)


ind_to_col_name_pp = partial(ind_to_col_name, posterior=False)


def fit_predict(
        norm_train_features, norm_test_features,
        observed, random_seeds=(42, 1)):
    """
    This function brings together the following operations:
    fit;
    predict the normalized target for train and test set;
    separate the predictions in 2 DataFrames,
        which correspond to the train and test sets.

        Args:
        norm_train_features (DataFrame): the features used to train the model.
        observed (DataFrame with 1 column): the target values.
        norm_test_features (DataFrame): the new data for applying the model.
        random_seed (tuple of ints, optional): the random seeds
            of fit and predict.
            Defaults to (42, 1).
    """

    idata = fit(norm_train_features, observed, random_seeds[0])
    pp = predict(
        idata, norm_train_features,
        norm_test_features, random_seeds[1])

    df_pp_full = pp.to_dataframe()  # posterior predictive (DataFrame)

    # predictions on train set
    train_size = norm_train_features.shape[0]
    lst_cols_train = list(map(ind_to_col_name_pp, range(train_size)))
    df_posterior_pred_train = df_pp_full[lst_cols_train]

    # predictions on test set
    test_size = norm_test_features.shape[0]
    lst_cols_test = list(map(
        ind_to_col_name_pp, range(train_size, train_size + test_size)))
    df_posterior_pred_test = df_pp_full[lst_cols_test]

    return df_posterior_pred_train, df_posterior_pred_test

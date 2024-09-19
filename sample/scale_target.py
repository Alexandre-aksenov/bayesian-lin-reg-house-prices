import numpy as np


def ScaleTarget(Y_train):
    """
    Normalize the log-norm-price variable, produce the function
    for converting a prediction back to real prices.

    This function is analogous to 'StandardScaler',
    but is customized to the needs of the present task such as:

    include the exponent for converting log-prices to the scale of real prices;
    be able to denormalize a Series or a DataFrame .

    Args:
        Y_train (Series): unnormalized log-prices.

    Returns:
        norm_Y_train (Series): train set of normalized log-prices,
        Sale_Price : inverse function
            predictions of normalized log-prices ->
            predictions of real prices.
    """

    mean_Y_train = Y_train.mean()
    std_Y_train = Y_train.std(ddof=0)
    norm_Y_train = (Y_train - mean_Y_train) / std_Y_train

    # define the inverse function
    def Sale_Price(normalized_log):
        """
        The inverse to the normalization operations above .
        normalized LogSalePrice (the target variable in the models)
        -> SalePrice in dollars.

        'log-sale-price' is computed "manually"
        so that the function can be applied
            either to a Series (in case of linear regression)
            or to a DataFrame  (in case of bayesian regression).

        Args:
            normalized_log (Series or DataFrame of floats):
            predicted normalized log-prices

        Returns:
            array of floats of the same size as input: the predicted prices.
    """

        log_sale_price = normalized_log * std_Y_train + mean_Y_train
        sale_Price = np.exp(log_sale_price) - 1
        return sale_Price

    return norm_Y_train, Sale_Price

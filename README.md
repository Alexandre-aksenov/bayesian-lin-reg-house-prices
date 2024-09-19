<b>About the dataset.</b>

The dataset comes from:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

<b>About the problem.</b>

Sale prices of houses are estimated using most correlated quantitative features. The prices are transformed to logarithmic scale, then normalized to bring them more similar to the hypotheses of the regression model. 5 quantitative features, which are most correlated to the target variable, are selected for regression.  

The script <code>EDA.ipynb</code> contains Exploratory Data Analysis: dealing with missing values, tranformation of the target variable to the log-scale, detection of the most correlated quantitative features. The script <code>train_test_split.ipynb</code> contains detailed estimation using both models on one particular train-test split. The script <code>stratified_kfold.ipynb</code> contains 70-fold Cross-Validation of both models. All presented splits are stratified on the discretized target variable. 

The module <code>scale_target.py</code> contains a function for:
* transforming the original target variable (price in dollars) into its scaled logathmic version,
* keeping the necessary data in memory for inverse transformation.

The module <code>bayesian_linear.py</code> contains a user-friendly interface for the bayesian linear model. It uses the library <code>pymc</code> as backend.

<b>Selected models.</b>

A model of bayesian linear regression is presented. Its performance is measured using MAE and compared to the baseline model of simple linear regression.

<b>Results.</b>

Although the intermediate results show that the estimates of coefficients are similar for both models, the conversion to the prices (mean of exponents of the predictive distribution for the Bayesian model versus exponent of the mean for the linear model) creates a difference between predictions.
Despite the MAE are quite different across splits (SD equal 10000$),
the error of the bayesian models tends to be smaller than MAE for the linear model (mean difference between errors equals 120$).
This slight improvement can be detected either at the level of the mean across folds, or at the level of differences between both errors for the same fold.

<b>Possible improvements.</b>

Another possible use of bayesian modeling can consist in moving the exponent inside the model, resulting in a Generalized Linear Model. A possible difficulty of this approach may consist in selecting appropriate distributions of error terms.

Another possible improvement, purely technical, may consist in converting the module <code>bayesian_linear.py</code> into a class to keep the intermediate result (posterior predictive distributions of coefficients) available after prediction.

<b>Feedback and additional questions.</b>

All questions about the source code should be adressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com

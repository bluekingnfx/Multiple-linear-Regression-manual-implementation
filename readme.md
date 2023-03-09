# Under the hood of Multiple Linear Regression (MLR)

**You can install requried modules from requirements.txt, Enter the following code command prompt**
```bash
pip install -r requirements.txt
```

### SVD not implemented so feature matrix with multi-collinear predictors will not give correct regression coefficients. In short, Multi-collinear predictors are not handled in the model, so please make sure the design matrix is not multi-collinear while using MUR. Soon SVD will be implemented.

### Currently the version is $\alpha$

### Files
 - MUR Implementation.iynb explains step by step process
 - Main.py is complete implementation of MUR without exploitation.

## **Formulas used**

> ### Regression Equation
>
> $y = b_0 + \sum^{n}_{i=1}{b_i \times x_i} + e$
>
> - $b_0$: Intercept of regression model
> 
> - $x_i$: $i^{th}$ feature(or predictor) of model
> 
> - $b_i$: Regression coefficient of $x_i$
> 
> - e is the error (Sum of squared error)

#### In multiple linear regression the coefficients are calculated by matrices

> ### Regression coefficients matrix.
> 
> $b = (X^TX)^{-1}X^TY$
> 
> - X is design matrix.
> 
> - $X^T$ is transpose of the design matrix
> 
> - Y is the Response matrix
> 
> - b is the **Regression coefficients matrix**

_**Definitions**_:

Design matrix is matrix of predictor columns prepended by the constant column

A transpose matrix is an array or matrix created by interchanging the rows and columns of an original matrix. It is denoted by $A^T$.

The inverse of a matrix is another matrix that, when multiplied by the given matrix, produces the identity matrix(Matrices of 0s and 1s).

$A^{-1}=\frac{1}{\det(A)}\text{adj}(A)$

$AA^{-1}=A^{-1}A=I$


### Finding p values of regression coefficients involves calculating T-Statistic  

> ### T-static of regression coefficients
> 
> $\sum_{i=0}^{n}{T_i = \frac{b_i}{S_{bi}}}$
>
> - $S_{bi}$ : Standard error of the slope $b_i$
> 
> ### $S_{b_i}$ of regression of coefficients
> 
> $S_{bi} = \sqrt{MSE * (XX^T)^{-1}}$
>
> - MSE: Mean square error or average of sum of residual error
> 
> - $(XX^T)^{-1}$: Covariance matrix
>
> - **Diagonal of covariance matrices are non-negative(or whole numbers) corresponds to the standard error of slopes**
>
> ### To calculate MSE
>
> MSE = $\frac{SSE}{(n - k - 1)}$
>
> - SSE: Sum of squared errors or Sum of squared **Residuals** or Variance of each point from its regression point.
>
> - n: No of samples (rows in the dataset)
>
> - k: No of features (Not including intercept)
>
> ### To calculate SSE (Sum of squared Residuals)
>
> SSE = $\sum_{i=0}^{n}{(y_i - \hat{y_i})}$
>
> - $y_i$ each ith(each) data point in the response variable
> 
> - $\hat{y_i}$ is the predicted value of $y_i$ obtained by substituting $x_i$ and $b_i$ in regression equation.

P values considering hypothesis testing 

$H_0$ : Population regression coefficient $(\beta)$ = 0 such that no linear relationship(as x varies y also varies)

$H_a$ : $\beta \ne 0$ suggesting as x varies y also varying

- Indicating Two tailed test from symbol of alternate hypothesis. 

- The difference in the p value of -b and b gives the total probability which tested against chosen significance level(say 0.05 or 5%). 


Example:

Say the t statistic of slope A is **-4.342** and degrees of freedom is 10

Need to find the probability associated with -4.342 and 10 DOF from t table or ``scipy.stats.t.cdf(-4.342,10)``which is the left tail(probability to the left). 

To find right tail one can just take ``1 - scipy.stats.t.cdf(4.342,10)``(probability to right) will give the right tail. To get p desired p value is ``scipy.stats.t.cdf(-4.342,10)`` + ``1 - scipy.stats.t.cdf(4.342,10)``

`scipy.stats.t.cdf(-4.342,10)` and `1 - scipy.stats.t.cdf(4.342,10)` will be same because the t-distribution is symmetrical.

Given the inference symmetrical, one can ``(1 - scipy.stats.t.cdf(abs(-4.342),10))*2``

```py
#Try
import scipy 
print(scipy.stats.t.cdf(-4.342,10) + (1- scipy.stats.t.cdf(4.342,10)),(1 - scipy.stats.t.cdf(abs(-4.342),10))*2)
```

> ### Calculation of $R^2$
> 
> $R^2 = 1 - \frac{SSE}{TSS}$
> 
> $R^2 = \frac{SSR}{TSS}$ 
>
> - TSS: Total sum of squares
> 
> - SSR: Sum of squares of Regression (or amount of variability in y explained by regression line)
>
> ### To calculate TSS
>
> $TSS = \sum_{i=0}^{n}{y_i - \bar{y}}$
>
> - $y_i$ is the ith data point of i and y is the mean of y. It explains the total variability of points explained by mean of y. (Worst case)
> 
> - $\bar{y}$ mean of y.
>
> ### To calculate the adjusted $R^2$
>
> $Adjusted \space R^2 = 1 - \frac{(n-2)(1 - R^2)}{n - k - 1}$
>
> - If any predictor in the model does not impact the model can addressed the adjacent r-squared
>
> - The adjusted R-squared value takes into account the number of predictors in the model and adjusts the R-squared value accordingly



# Main File Usage (MUR.py) 

### To compute the Multiple linear regression and to get all values 

```py
from MUR import MUR
MUR(X,y).summary #X is the Features Data frame, y is pandas.series includes r squared and adjusted r squared.
```

### To get the Regression coefficients alone 

```py
from MUR import MUR
MUR.RegressionCoffMatrix(X,y) #X is the Features Data frame, y is pandas.series
```
Gives back dictionary comprising key as feature names and regression coefficients as values (Including intercept).


### To get the Standard errors of regression coefficient.

```py
from MUR import MUR
MUR.StandardErr(X,y) #X is the Features Data frame, y is pandas.series
```
Returns Standard error of regression coefficients and regression coefficients of associated features as a dictionary (Including intercept).


### To get T statistic and p values

```py
from MUR import MUR
MUR.TStatisticWithPValues(X,y) #X is the Features Data frame, y is pandas.series
```

Returns T-statistic,p-values, Standard error of regression coefficients and regression coefficients of associated features as a dictionary.


### To get y_pred and y to compare.

```py
from MUR import MUR
MUR.predictYValues(X,y) #X is the Features Data frame, y is pandas.series
```

Returns Predicted values of y and y in a data frame to compare.

### To get an df with constant column to your data frame.

```py
from MUR import MUR
MUR.attach_ConstantCols(X,y)
```
It takes design matrix or X_train in context attaches ones columns to it.


### Things to keep in mind.

- Make sure that the X (feature matrix) does not have **collinear predictors among themselves (no Multi-collinear predictors)** before using MUR class.

- Don't the default values of ``raw`` and ``Regression_Coff`` to anything as they are *__reserved to internal use__*. They left intentionally to give sense what happens internally.
- If you got feature named **Const** in your predictor X (features data frame) change the feature name to something else.

Made with ❤️
____


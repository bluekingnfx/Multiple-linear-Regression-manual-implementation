import numpy as np
import pandas as pd
from scipy.stats import t


class MUR():
    
    @classmethod
    def attach_ConstantCols(_,X:pd.DataFrame) -> pd.DataFrame:
        """
        #### You can get entire summary by summary property

        >>> import MUR as mu
        >>> mu.MUR(X,y).summary

        #### If you interested to get parts of results you can use the class methods of MUR Class.

        It takes design matrix or X_train in context attaches ones columns to it.

        # Args 
            - X : Pandas data frame

        # Returns 
            - X_with_const_cols: Pandas data frame
        
        ----

        # Example 
        
        Say X is the dataset(table) below
        
        |        Name | Age           | Gender |
        |-------------|---------------|--------|
        | John Smith  | 25            | Male   |
        | Jane Doe    | 30            | Female |
        | Bob Johnson | 45            | Male   |

        ```py
        >>> MUR.attach_ConstantCols(X)
        ```
        
        Gives back:
        |  const | Name         | Age | Gender |
        |--------|--------------|-----|--------|
        | 1      | John Smith   | 25  | Male   |
        | 1      | Jane Doe     | 30  | Female |
        | 1      | Bob Johnson  | 45  | Male   |


        """
        X_with_ones_cols = X.copy()
        X_with_ones_cols.insert(loc=0,column="Const",value=np.ones((X.shape[0],1),float))
        return X_with_ones_cols

    @classmethod
    def RegressionCoffMatrix(cls,X:pd.DataFrame,y:pd.Series,raw=True) -> dict:

        """
        #### You can get entire summary by summary property

        >>> import MUR as mu
        >>> mu.MUR(X,y).summary

        #### If you interested to get parts of results you can use the class methods of MUR Class.

        # Args

        X: Features Data frame.

        y: Response column or Target column **(pd.series)**

        raw: default True (Reserved for internal use)

        Disabling causes un excepted results

        # Returns
        Dictionary comprising key as feature names and regression coefficients as values (Including intercept)
        
        # Example 

        >>> X

        Say X is the dataset(table) below
        
        | Age | Gender | Experience  | Prof_Finance | Prof_IT | Prof_Marketing |
        |-----|--------|-------------|--------------|---------|----------------|
        | 25  | 0      | 3           | 0            | 0       | 1              |
        | 30  | 1      | 5           | 0            | 0       | 0              |
        | 45  | 0      | 10          | 1            | 0       | 0              |
        | 22  | 1      | 1           | 0            | 0       | 1              |
        | 35  | 1      | 8           | 0            | 1       | 0              |
        | 42  | 0      | 15          | 0            | 1       | 0              |        
        | 27  | 1      | 4           | 1            | 0       | 0              |
        | 39  | 0      | 7           | 0            | 0       | 0              |
        | 50  | 1      | 12          | 0            | 0       | 0              |
        >>> y

        | Salary |
        |--------|
        | 60000  |
        | 75000  |
        | 100000 |
        | 55000  |
        | 85000  |
        | 90000  |
        | 72000  |
        | 65000  |
        | 110000 |

        >>> print(RegressionCoffMatrix(X,y))
        {'Age': 5432.7534898049225,
        'Experience': 14060.377358490568,
        'Gender': 1299.0888173030155,
        'Prof_Finance': 2124.5589814389996,
        'Prof_IT': 11897.94753796603,
        'Prof_Marketing': 589.7100782325706}
        

        """


        if raw == True:
            colNames = ["Const",*X.columns.to_list()]
            X_Cols = (cls.attach_ConstantCols(X)).to_numpy()
        else:
            colNames = X.columns
            X_Cols = X.to_numpy()

        y = pd.DataFrame(y)

        XT = np.transpose(X_Cols)
        XT_X = np.matmul(XT,X_Cols)
        Inv_XT_X = np.linalg.inv(XT_X)
        Info_Matrix = np.matmul(Inv_XT_X,XT)
        reg_matrix = np.matmul(Info_Matrix,y.to_numpy())
        dict_RegCoff = {}

        for ind,i in enumerate(colNames):
            
            if i!="Const":
                dict_RegCoff = {**dict_RegCoff,i:reg_matrix.flatten()[ind]}
            else:
                dict_RegCoff = {**dict_RegCoff,"Intercept_or_B0": reg_matrix.flatten()[ind]}

        return dict_RegCoff
    
    @classmethod
    def StandardErr(cls,X:pd.DataFrame,y:pd.Series,Regression_Coff = {},raw=True) -> dict:

        """

            #### You can get entire summary by summary property

            >>> import MUR as mu
            >>> mu.MUR(X,y).summary

            #### If you interested to get parts of results you can use the class methods of MUR Class.

            # Args

            X: Features Data frame
            
            Y: Target series (Response variable)

            Regression_coff: It is reserved for internal use. Misusing Regression_Coff cause unexpected results

            raw: It is reserved too. Don't change to False.


            # Returns
            
            Returns Standard error of regression coefficients and regression coefficients of associated features as a dictionary.

            # Example

            >>> X

            Say X is the dataset(table) below
            
            | Age | Gender | Experience  | Prof_Finance | Prof_IT | Prof_Marketing |
            |-----|--------|-------------|--------------|---------|----------------|
            | 25  | 0      | 3           | 0            | 0       | 1              |
            | 30  | 1      | 5           | 0            | 0       | 0              |
            | 45  | 0      | 10          | 1            | 0       | 0              |
            | 22  | 1      | 1           | 0            | 0       | 1              |
            | 35  | 1      | 8           | 0            | 1       | 0              |
            | 42  | 0      | 15          | 0            | 1       | 0              |        
            | 27  | 1      | 4           | 1            | 0       | 0              |
            | 39  | 0      | 7           | 0            | 0       | 0              |
            | 50  | 1      | 12          | 0            | 0       | 0              |
            >>> y

            | Salary |
            |--------|
            | 60000  |
            | 75000  |
            | 100000 |
            | 55000  |
            | 85000  |
            | 90000  |
            | 72000  |
            | 65000  |
            | 110000 |

            >>> print(MUR.StandardErr(X,y))
            [{'FeatureName': 'Intercept_or_B0',
            'RegCoff': 5432.7534898049225,
            'StdErr': 28499.757752251477},
            {'FeatureName': 'Age',
            'RegCoff': 1299.0888173030155,
            'StdErr': 1140.0546226776412},
            {'FeatureName': 'Gender',
            'RegCoff': 14060.377358490568,
            'StdErr': 7158.152039823275},
            {'FeatureName': 'Experience',
            'RegCoff': 2124.5589814389996,
            'StdErr': 2809.0006393266167},
            {'FeatureName': 'Prof_Finance',
            'RegCoff': 11897.94753796603,
            'StdErr': 8578.820845047967},
            {'FeatureName': 'Prof_IT',
            'RegCoff': 589.7100782325706,
            'StdErr': 13511.416220700361},
            {'FeatureName': 'Prof_Marketing',
            'RegCoff': 10259.352661451523,
            'StdErr': 12014.199003728838}]
    
            """
    

        if raw == True:
            cls._checkErr1(X,y,Regression_Coff)
            X = cls.attach_ConstantCols(X)
            Regression_Coff = cls.RegressionCoffMatrix(X,y,raw=False)
    
        y = pd.DataFrame(y)
        y_pred = cls.predictYValues(X,y,Regression_Coff=Regression_Coff,raw=False)
        SSR = cls._SSR(y.to_numpy(),y_pred.to_numpy())
        MSR = SSR/(X.shape[0] - (X.shape[1] - 1) - 1)
        S_b = np.sqrt(cls._nonSquaredRegCoff(X.to_numpy(),MSR))
        if raw == True:
            return [{"FeatureName": i,"RegCoff":Regression_Coff[i],"StdErr": S_b[ind]} for ind,i in enumerate(Regression_Coff)]
        else:
            return {"S_b":S_b,"y_pred":y_pred,"SSR":SSR}
        
    @classmethod
    def TStatisticWithPValues(cls,X:pd.DataFrame,y:pd.Series,raw = True,Regression_Coff = {}) -> dict:

        """

            #### You can get entire summary by summary property

            >>> import MUR as mu
            >>> mu.MUR(X,y).summary

            #### If you interested to get parts of results you can use the class methods of MUR Class.

            # Args

            X: Features Data frame
            
            Y: Target series (Response variable)

            Regression_coff: It is reserved for internal use. Misusing Regression_Coff cause unexpected results

            raw: It is reserved too. Don't change to False.


            # Returns
            
            Returns T-statistic,p-values, Standard error of regression coefficients and regression coefficients of associated features as a dictionary.

            # Example

            >>> X

            Say X is the dataset(table) below
            
            | Age | Gender | Experience  | Prof_Finance | Prof_IT | Prof_Marketing |
            |-----|--------|-------------|--------------|---------|----------------|
            | 25  | 0      | 3           | 0            | 0       | 1              |
            | 30  | 1      | 5           | 0            | 0       | 0              |
            | 45  | 0      | 10          | 1            | 0       | 0              |
            | 22  | 1      | 1           | 0            | 0       | 1              |
            | 35  | 1      | 8           | 0            | 1       | 0              |
            | 42  | 0      | 15          | 0            | 1       | 0              |        
            | 27  | 1      | 4           | 1            | 0       | 0              |
            | 39  | 0      | 7           | 0            | 0       | 0              |
            | 50  | 1      | 12          | 0            | 0       | 0              |
            >>> y

            | Salary |
            |--------|
            | 60000  |
            | 75000  |
            | 100000 |
            | 55000  |
            | 85000  |
            | 90000  |
            | 72000  |
            | 65000  |
            | 110000 |

            >>> print(MUR.TStatisticWithPValues(X,y))
            [{'FeatureName': 'Intercept_or_B0',
            'RegCoff': 5432.7534898049225,
            'StdErr': 28499.757752251477,
            'TStatistic': 0.19062454976045318,
            'p-value': 0.8664161586109389},
            {'FeatureName': 'Age',
            'RegCoff': 1299.0888173030155,
            'StdErr': 1140.0546226776412,
            'TStatistic': 1.139496995549083,
            'p-value': 0.37258050645568774},
            {'FeatureName': 'Gender',
            'StdErr': 8578.820845047967,
            'TStatistic': 1.3868977745157127,
            'p-value': 0.29982205169647624},
            {'FeatureName': 'Prof_IT',
            'RegCoff': 589.7100782325706,
            'StdErr': 13511.416220700361,
            'TStatistic': 0.043645319528318335,
            'p-value': 0.9691527854202489},
            {'FeatureName': 'Prof_Marketing',
            'RegCoff': 10259.352661451523,
            'StdErr': 12014.199003728838,
            'TStatistic': 0.8539356355148883,
            'p-value': 0.48309947506516315}]
                
            """
    

        if raw == True:
            cls._checkErr1(X,y,Regression_Coff)
            X = cls.attach_ConstantCols(X)
            Regression_Coff = cls.RegressionCoffMatrix(X,y,raw=False)
        
        
        S_bYPredSSR = cls.StandardErr(Regression_Coff=Regression_Coff,X=X,y=y,raw=False)

        StdSlopeRigDic = [{"FeatureName": i,"RegCoff":Regression_Coff[i],"StdErr": S_bYPredSSR["S_b"][ind],"TStatistic":(Regression_Coff[i]/S_bYPredSSR['S_b'][ind]),"p-value":(1 - t.cdf((abs(Regression_Coff[i]/S_bYPredSSR['S_b'][ind])),(X.shape[0] - (len(X.columns) - 1) - 1)))*2} for ind,i in enumerate(Regression_Coff)]

        return StdSlopeRigDic

    @classmethod 
    def predictYValues(cls,X:pd.DataFrame,y:pd.Series,raw = True,Regression_Coff = {}) -> pd.DataFrame:

        """

            #### You can get entire summary by summary property

            >>> import MUR as mu
            >>> mu.MUR(X,y).summary

            #### If you interested to get parts of results you can use the class methods of MUR Class.

            # Args

            X: Features Data frame
            
            Y: Target series (Response variable)

            Regression_coff: It is reserved for internal use. Misusing Regression_Coff cause unexpected results

            raw: It is reserved too. Don't change to False.


            # Returns
            
            Returns Predicted values of y and y in a data frame to compare.

            # Example

            >>> X

            Say X is the dataset(table) below
            
            | Age | Gender | Experience  | Prof_Finance | Prof_IT | Prof_Marketing |
            |-----|--------|-------------|--------------|---------|----------------|
            | 25  | 0      | 3           | 0            | 0       | 1              |
            | 30  | 1      | 5           | 0            | 0       | 0              |
            | 45  | 0      | 10          | 1            | 0       | 0              |
            | 22  | 1      | 1           | 0            | 0       | 1              |
            | 35  | 1      | 8           | 0            | 1       | 0              |
            | 42  | 0      | 15          | 0            | 1       | 0              |        
            | 27  | 1      | 4           | 1            | 0       | 0              |
            | 39  | 0      | 7           | 0            | 0       | 0              |
            | 50  | 1      | 12          | 0            | 0       | 0              |
            >>> y

            | Salary |
            |--------|
            | 60000  |
            | 75000  |
            | 100000 |
            | 55000  |
            | 85000  |
            | 90000  |
            | 72000  |
            | 65000  |
            | 110000 |

            >>> print(MUR.predictYValues(X,y))

            | Index | Salary | Y predicted   |
            |-------|--------|---------------|
            | 0     | 60000  | 54543.003528  |
            | 1     | 75000  | 69088.590275  |
            | 2     | 100000 | 97035.287621  |
            | 3     | 55000  | 60456.996472  |
            | 4     | 85000  | 82547.421384  |
            | 5     | 90000  | 92452.578616  |
            | 6     | 72000  | 74964.712379  |
            | 7     | 65000  | 70969.130235  |
            | 8     | 110000 | 109942.279491 |
        """

        if raw == True:
            cls._checkErr1(X,y,Regression_Coff)
            X = cls.attach_ConstantCols(X)
            Regression_Coff = cls.RegressionCoffMatrix(X,y,raw=False)
        
        b_Matrix = np.reshape(list(Regression_Coff.values()),(X.shape[1],1))

        y_pred = np.matmul(X,b_Matrix)
        if raw == True:
            y_pred_df = pd.DataFrame(y_pred)
            y_pred_df.columns = ["Y predicted"]
            return pd.concat([y,y_pred_df],axis=1)
        return y_pred

    @classmethod
    def _R_SqAndRAdj(cls,SSR:np.float64,y:pd.DataFrame,n:int,k:int):
        TSS = cls._TSS(y.to_numpy())
        R_sq = 1 - (SSR/TSS)
        AdjR_sq = 1 - ((n-1)*(1 - R_sq))/(n - k - 1)
        return {
            "R_sq":R_sq,
            "AdjR_sq":AdjR_sq
        }

    @classmethod
    def _TSS(cls,y:np.ndarray) -> float | int:
        return np.sum((y- np.mean(y))**2)
    
    @classmethod
    def _SSR(cls,y:np.ndarray,y_pred:np.ndarray):
        return np.sum((y_pred - y)**2)

    @classmethod
    def _nonSquaredRegCoff(cls,X,MSR): 

        X_transpose = np.transpose(X)
        Inv_Trans = np.linalg.inv(np.matmul(X_transpose,X))
        return np.array([i[ind] for ind,i in enumerate(MSR*Inv_Trans)])

    @classmethod
    def _checkErr1(cls,X,y,dictTocheck):
        if 0 in X.shape or 0 in y.shape:
            raise Exception("Pass a valid data frame. either of x or y has row or column equal to 0.")
        if len(dictTocheck) != 0:
            raise Exception("You are trying to change the reserved argument.")

    def __init__(self,X:pd.DataFrame,y:pd.DataFrame) -> None:
        """
        Computes Multiple linear regression and gives back vital information about the dataset.

        ### If there is feature called Const in your dataset please rename it to something else. 
        
        # Args

        X: Features Data frame
        
        Y: Target pandas series (Response variable)

        # Returns 

        Gives back vital information about the dataset helps to infer the linear relationship with features and target variable.

        # Example

        >>> X

        Say X is the dataset(table) below
        
        | Age | Gender | Experience  | Prof_Finance | Prof_IT | Prof_Marketing |
        |-----|--------|-------------|--------------|---------|----------------|
        | 25  | 0      | 3           | 0            | 0       | 1              |
        | 30  | 1      | 5           | 0            | 0       | 0              |
        | 45  | 0      | 10          | 1            | 0       | 0              |
        | 22  | 1      | 1           | 0            | 0       | 1              |
        | 35  | 1      | 8           | 0            | 1       | 0              |
        | 42  | 0      | 15          | 0            | 1       | 0              |        
        | 27  | 1      | 4           | 1            | 0       | 0              |
        | 39  | 0      | 7           | 0            | 0       | 0              |
        | 50  | 1      | 12          | 0            | 0       | 0              |
        >>> y

        | Salary |
        |--------|
        | 60000  |
        | 75000  |
        | 100000 |
        | 55000  |
        | 85000  |
        | 90000  |
        | 72000  |
        | 65000  |
        | 110000 |

        >>> import MUR as mu
        >>> mu.MUR(X,y).summary
        {
            'Feature_wise_info': [
                {
                    'FeatureName': 'Intercept_or_B0',
                    'RegCoff': 5432.7534898049225,
                    'StdErr': 28499.757752251477,
                    'TStatistic': 0.19062454976045318,
                    'p-value': 0.8664161586109389
                },
                {
                    'FeatureName': 'Age',
                    'RegCoff': 1299.0888173030155,
                    'StdErr': 1140.0546226776412,
                    'TStatistic': 1.139496995549083,
                    'p-value': 0.37258050645568774
                },
                {
                    'FeatureName': 'Gender',
                    'RegCoff': 14060.377358490568,
                    'StdErr': 7158.152039823275,
                    'TStatistic': 1.9642468168135891,
                    'p-value': 0.18845700994396797
                },
                {
                    'FeatureName': 'Experience',
                    'RegCoff': 2124.5589814389996,
                    'StdErr': 2809.0006393266167,
                    'TStatistic': 0.7563397998899374,
                    'p-value': 0.5283962389969279
                },
                {
                    'FeatureName': 'Prof_Finance',
                    'RegCoff': 11897.94753796603,
                    'StdErr': 8578.820845047967,
                    'TStatistic': 1.3868977745157127,
                    'p-value': 0.29982205169647624
                },
                {
                    'FeatureName': 'Prof_IT',
                    'RegCoff': 589.7100782325706,
                    'StdErr': 13511.416220700361,
                    'TStatistic': 0.043645319528318335,
                    'p-value': 0.9691527854202489
                },
                {
                    'FeatureName': 'Prof_Marketing',
                    'RegCoff': 10259.352661451523,
                    'StdErr': 12014.199003728838,
                    'TStatistic': 0.8539356355148883,
                    'p-value': 0.48309947506516315
                }
            ],
            Overall_model_info': {
                'AdjR_sq': 0.7682234395289487,
                R_sq': 0.9420558598822372
            }
        }

        ----
        
        ❤️ from Blueking NFX
        """
        self.X = X.copy()
        self.y = pd.DataFrame(y.copy())
        X_with_ones_col = self.attach_ConstantCols(self.X)
        RegressionCoffMatrix = self.RegressionCoffMatrix(X=X_with_ones_col,y=self.y,raw=False)
        StandardErr = self.StandardErr(X_with_ones_col,self.y,RegressionCoffMatrix,raw=False)
        CoEffRInfo = self._R_SqAndRAdj(StandardErr["SSR"],self.y,len(self.y),(len(X.columns)),)
        TStatisticWithPValues = self.TStatisticWithPValues(X_with_ones_col,self.y,Regression_Coff=RegressionCoffMatrix,raw=False)
        self.summary = {
            "Overall_model_info":CoEffRInfo,
            "Feature_wise_info":TStatisticWithPValues
        }

if __name__ == "__main__":
    np.random.seed(10)
    XFrame = pd.DataFrame(np.random.randint(1,20,(100,2)))
    XFrame.columns = ["A","B"]
    YFrame = pd.DataFrame(np.random.randint(1,100,(100,1)))
    YFrame.columns = ["Y"]


    from pprint import pprint

    pprint(MUR(XFrame,YFrame).summary)
    import statsmodels.api as sm
    print(sm.OLS(YFrame,sm.add_constant(XFrame)).fit().summary())
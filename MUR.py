import numpy as np
import pandas as pd
from scipy.stats import t


class MUR():
    
    @classmethod
    def attach_ConstantCols(_,X:pd.DataFrame) -> pd.DataFrame:
        """
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
        X_with_ones_cols = X
        X_with_ones_cols.insert(loc=0,column="Const",value=np.ones((X.shape[0],1),int))
        return X_with_ones_cols

    @classmethod
    def RegressionCoffMatrix(cls,X_with_ones_col:pd.DataFrame,y:pd.DataFrame,raw=False) -> dict:

        """
        Returns Regression coefficients dictionary.

        # Args

        X_with_ones_cols: X with ones column to calculate const slope along with other features regression coefficients.

        y: Response column or Target column

        raw: default False (Optional)

        By Enabling you can pass X and Y (Raw data Frame) Internal class methods take care about previous steps.

        # Returns
        Dictionary comprising key as feature names and regression coefficients as values (Including intercept)
        
        """
        if raw == True:
            X = (cls.attach_ConstantCols(X_with_ones_col)).to_numpy()

        else:
            X = X_with_ones_col.to_numpy()

        XT = np.transpose(X)
        XT_X = np.matmul(XT,X)
        Inv_XT_X = np.linalg.inv(XT_X)
        Info_Matrix = np.matmul(Inv_XT_X,XT)
        reg_matrix = np.matmul(Info_Matrix,y.to_numpy())
        dict_RegCoff = {}

        for ind,i in enumerate(X_with_ones_col.columns):
            
            if i!="Const":
                dict_RegCoff = {**dict_RegCoff,i:reg_matrix.flatten()[ind]}
            else:
                dict_RegCoff = {**dict_RegCoff,"Intercept_or_B0": reg_matrix.flatten()[ind]}

        return dict_RegCoff
    
    @classmethod
    def StandardErr(cls,raw=False,Regression_Coff = {},X:pd.DataFrame=pd.DataFrame({}),y:pd.DataFrame=pd.DataFrame({})):
        if raw == True:
            cls._checkErr1(X,y,Regression_Coff)
            X = cls.attach_ConstantCols(X)
            Regression_Coff = cls.RegressionCoffMatrix(X,y)
    

        y_pred = cls.predictYValues(y,X,Regression_Coff=Regression_Coff)
        SSR = cls._SSR(y.to_numpy(),y_pred)
        MSR = SSR/(X.shape[0] - (X.shape[1] - 1) - 1)
        S_b = np.sqrt(cls._nonSquaredRegCoff(X.to_numpy(),MSR))
        if raw == True:
            return [{"FeatureName": i,"RegCoff":Regression_Coff[i],"StdErr": S_b[ind]} for ind,i in enumerate(Regression_Coff)]
        else:
            return S_b
        

    @classmethod
    def t_statisticNStandardErr(cls,raw = False,Regression_Coff = {},X:pd.DataFrame=pd.DataFrame({}),y:pd.DataFrame=pd.DataFrame({})):

        if raw == True:
            cls._checkErr1(X,y,Regression_Coff)
            X = cls.attach_ConstantCols(X)
            Regression_Coff = cls.RegressionCoffMatrix(X,y)
        
        stdErrOfSlopes = cls.StandardErr(Regression_Coff=Regression_Coff,X=X,y=y)

        StdSlopeRigDic = [{"FeatureName": i,"RegCoff":Regression_Coff[i],"StdErr": stdErrOfSlopes[ind],"TStatistic":(Regression_Coff[i]/stdErrOfSlopes[ind])} for ind,i in enumerate(Regression_Coff)]

        return StdSlopeRigDic

    @classmethod
    def _SSR(cls,y,y_pred):
        return sum((y_pred.to_numpy() - y)**2)
    
    @classmethod
    def _nonSquaredRegCoff(cls,X,MSR):
        X_transpose = np.transpose(X)
        Inv_Trans = np.linalg.inv(np.matmul(X_transpose,X))
        return np.array([i[ind] for ind,i in enumerate(MSR*Inv_Trans)])

    @classmethod 
    def predictYValues(cls,y:pd.DataFrame,X:pd.DataFrame,raw = False,Regression_Coff = {}):
        if raw == True:
            cls._checkErr1(X,y,Regression_Coff)
            X = cls.attach_ConstantCols(X)
            Regression_Coff = cls.RegressionCoffMatrix(X,y)
        
        b_Matrix = np.reshape(list(Regression_Coff.values()),(X.shape[1],1))
        
        y_pred = np.matmul(X,b_Matrix)
        if raw == True:
            y_pred_df = pd.DataFrame(y_pred)
            y_pred_df.columns = ["Y predicted"]
            return pd.concat([y,y_pred_df],axis=1)
        return y_pred

    @classmethod
    def _checkErr1(cls,X,y,dictTocheck):
        if 0 in X.shape or 0 in y.shape:
            raise Exception("Pass a valid data frame. either of x or y has row or column equal to 0.")
        if len(dictTocheck) != 0:
            raise Exception("If you have Regression coefficient dictionary just pass that by switching raw to False")


    def __init__(self,X:pd.DataFrame,y:pd.DataFrame) -> None:
        self.X = X
        self.y = y
        X_with_ones_col = self.attach_ConstantCols(X)
    


if __name__ == "__main__":
    np.random.seed(10)
    XFrame = pd.DataFrame(np.random.randint(1,20,(100,2)))
    XFrame.columns = ["A","B"]
    YFrame = pd.DataFrame(np.random.randint(1,100,(100,1)))
    YFrame.columns = ["Y"]
    df = pd.concat([XFrame,YFrame],axis=1)
    print(MUR.predictYValues(raw=True,X=XFrame,y=YFrame))
    #import statsmodels.api as sm
    #print(sm.OLS(YFrame,sm.add_constant(XFrame)).fit().summary())
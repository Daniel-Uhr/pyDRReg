import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import resample
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings

# Function to estimate ATE using Outcome Regression
def OR_ate(df, X_cols, T_col, Y_col):
    X = df[X_cols].values
    T = df[T_col].values
    Y = df[Y_col].values
    
    # Model for untreated (D=0)
    model_0 = LinearRegression().fit(X[T == 0], Y[T == 0])
    mu0 = model_0.predict(X)
    
    # Model for treated (D=1)
    model_1 = LinearRegression().fit(X[T == 1], Y[T == 1])
    mu1 = model_1.predict(X)
    
    # Estimate ATE
    OR_ate_estimate = np.mean(mu1 - mu0)
    
    # Calculate standard error using the treated and untreated groups' variances
    se_ate = np.sqrt(np.var(mu1 - mu0) / len(mu1))
    
    # Calculate confidence interval and p-value
    z_value = OR_ate_estimate / se_ate
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_value)))
    ci_lower = OR_ate_estimate - 1.96 * se_ate
    ci_upper = OR_ate_estimate + 1.96 * se_ate
    
    return {
        'Estimate': OR_ate_estimate,
        'SE': se_ate,
        't-stat': z_value,
        'p-value': p_value,
        'CI': (ci_lower, ci_upper)
    }

# Function to estimate ATT using Outcome Regression
def OR_att(df, X_cols, T_col, Y_col):
    # Separate treated (D=1) and untreated (D=0) data
    X_treated = df[df[T_col] == 1][X_cols].values
    Y_treated = df[df[T_col] == 1][Y_col].values
    X_control = df[df[T_col] == 0][X_cols].values
    Y_control = df[df[T_col] == 0][Y_col].values
    
    # Fit linear regression models
    model_treated = LinearRegression().fit(X_treated, Y_treated)
    model_control = LinearRegression().fit(X_control, Y_control)
    
    # Predict outcomes for treated and counterfactuals using control group model
    mu1_X = model_treated.predict(X_treated)
    mu0_X = model_control.predict(X_treated)  # Counterfactual for treated
    
    # Calculate deviations for treated
    deviations_treated = Y_treated - mu1_X
    
    # Mean of deviations
    deviations_mean = deviations_treated.mean()
    
    # Estimate ATT
    OR_att_estimate = (mu1_X.mean() - mu0_X.mean()) + deviations_mean
    
    # Calculate standard error
    se_att = np.sqrt(np.var(mu1_X - mu0_X) / len(mu1_X))
    
    # Calculate confidence interval and p-value
    z_value = OR_att_estimate / se_att
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_value)))
    ci_lower = OR_att_estimate - 1.96 * se_att
    ci_upper = OR_att_estimate + 1.96 * se_att
    
    return {
        'Estimate': OR_att_estimate,
        'SE': se_att,
        't-stat': z_value,
        'p-value': p_value,
        'CI': (ci_lower, ci_upper)
    }

# Function to estimate ATE using IPW (Inverse Probability Weighting)
def IPW_ate(df, X_cols, T_col, Y_col):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        formula_pscore = f"{T_col} ~ " + " + ".join(X_cols)
        df['pscore'] = smf.logit(formula_pscore, data=df).fit(disp=0).predict()
    
    df['W1'] = 1 / df['pscore']
    df.loc[df[T_col] == 0, 'W1'] = 0
    df['W2'] = 1 / (1 - df['pscore'])
    df.loc[df[T_col] == 1, 'W2'] = 0
    df['W_ATE'] = df['W1'] + df['W2']
    
    model_ate = sm.WLS(df[Y_col], sm.add_constant(df[T_col]), weights=df['W_ATE']).fit()
    
    return {
        'Estimate': model_ate.params[T_col],
        'SE': model_ate.bse[T_col],
        't-stat': model_ate.tvalues[T_col],
        'p-value': model_ate.pvalues[T_col],
        'CI': (model_ate.conf_int().loc[T_col, 0], model_ate.conf_int().loc[T_col, 1])
    }

# Function to estimate ATT using IPW (Inverse Probability Weighting)
def IPW_att(df, X_cols, T_col, Y_col):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        formula_pscore = f"{T_col} ~ " + " + ".join(X_cols)
        df['pscore'] = smf.logit(formula_pscore, data=df).fit(disp=0).predict()
    
    df['W_ATT'] = df['pscore'] / (1 - df['pscore'])
    df.loc[df[T_col] == 1, 'W_ATT'] = 1
    
    model_att = sm.WLS(df[Y_col], sm.add_constant(df[T_col]), weights=df['W_ATT']).fit()
    
    return {
        'Estimate': model_att.params[T_col],
        'SE': model_att.bse[T_col],
        't-stat': model_att.tvalues[T_col],
        'p-value': model_att.pvalues[T_col],
        'CI': (model_att.conf_int().loc[T_col, 0], model_att.conf_int().loc[T_col, 1])
    }

# Function to estimate ATE and ATT using Doubly Robust Estimator
def DR_ate_att(df, X_cols, T_col, Y_col):
    X_np = df[X_cols].values
    T_np = df[T_col].values
    Y_np = df[Y_col].values

    ps = LogisticRegression(C=1e6, max_iter=1000).fit(X_np, T_np).predict_proba(X_np)[:, 1]
    df["ps"] = ps

    mu_model = LinearRegression().fit(df[X_cols + [T_col]], df[Y_col])
    mu0 = mu_model.predict(df[X_cols].assign(**{T_col: 0}))
    mu1 = mu_model.predict(df[X_cols].assign(**{T_col: 1}))

    dr_ate = mu1 - mu0 + (T_np / ps) * (Y_np - mu1) - ((1 - T_np) / (1 - ps)) * (Y_np - mu0)
    dr_att = mu1 - mu0 + df[T_col] * (Y_np - mu1) - (1 - df[T_col]) * ps / (1 - ps) * (Y_np - mu0)

    return {
        'ATE_Estimate': np.mean(dr_ate),
        'ATT_Estimate': np.mean(dr_att)
    }

# Main class to perform estimation with various estimators
class pyDRReg:
    def __init__(self, df, X_cols, T_col, Y_col, method='ate', estimator='OR', n_bootstrap=50):
        self.df = df
        self.X_cols = X_cols
        self.T_col = T_col
        self.Y_col = Y_col
        self.method = method
        self.estimator = estimator.upper()
        self.n_bootstrap = n_bootstrap
        self.results = None
        self._run_estimation()
    
    def _select_estimator(self):
        if self.estimator == 'OR':
            return OR_ate if self.method == 'ate' else OR_att
        elif self.estimator == 'IPW':
            return IPW_ate if self.method == 'ate' else IPW_att
        elif self.estimator == 'DR':
            return DR_ate_att
        else:
            raise ValueError(f"Estimator '{self.estimator}' not recognized. Available estimators: 'OR', 'IPW', 'DR'.")

    def _run_estimation(self):
        estimates = []
        estimator_func = self._select_estimator()
        
        for _ in range(self.n_bootstrap):
            df_resampled = resample(self.df, replace=True, n_samples=len(self.df))
            
            if self.estimator == 'DR':
                dr_results = estimator_func(df_resampled, self.X_cols, self.T_col, self.Y_col)
                estimate = dr_results['ATE_Estimate'] if self.method == 'ate' else dr_results['ATT_Estimate']
            else:
                # Handle OR and IPW estimators correctly as dictionaries or scalars
                result = estimator_func(df_resampled, self.X_cols, self.T_col, self.Y_col)
                estimate = result['Estimate'] if isinstance(result, dict) else result
            
            estimates.append(estimate)
        
        # Calculate standard error, confidence intervals, and p-value
        se = np.std(estimates, ddof=1)
        mean_estimate = np.mean(estimates)
        ci_lower = mean_estimate - 1.96 * se
        ci_upper = mean_estimate + 1.96 * se
        z_value = mean_estimate / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_value)))
        
        # Store results
        self.results = {
            'Estimator': self.estimator,
            'Method': self.method.upper(),
            'Estimate': mean_estimate,
            'SE': se,
            't-stat': z_value,
            'p-value': p_value,
            'CI': (ci_lower, ci_upper)
        }
    
    def summary(self):
        if self.results is None:
            raise ValueError("Estimation has not been completed.")
        
        # Format results into a DataFrame for friendly display
        results_df = pd.DataFrame({
            'Metric': ['Estimator', 'Method', 'Estimate', 'SE', 't-stat', 'p-value', 'CI Lower', 'CI Upper'],
            'Value': [
                self.results['Estimator'],
                self.results['Method'],
                self.results['Estimate'],
                self.results['SE'],
                self.results['t-stat'],
                self.results['p-value'],
                self.results['CI'][0],  # CI Lower
                self.results['CI'][1]   # CI Upper
            ]
        })
        
        # Return the formatted DataFrame
        return results_df



# Load packages:
    
import numpy as np
from timeit import default_timer as timer#For timing
import random# For random number generation

from scipy.stats import norm, truncnorm, pareto, expon, uniform, binom# Generate from distributions: normal; truncated normal; powerlaw; exponential; uniform
from scipy.stats import rankdata#  For calculating a variant of the ecfd
from scipy.stats import skew, kurtosis#Skewness, kurtosis
from scipy.stats import hmean, gmean, iqr#Harmonic mean, geometric mean, interquartile range

from scipy.optimize import minimize, root# For fitting pdfs

import matplotlib.pyplot as plt# For plotting
import json#Saving and loading data

from os.path import exists#Check whether a file or directory exists
import csv#Read csv data

from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter, NullLocator, FuncFormatter, PercentFormatter, ScalarFormatter, LogFormatterSciNotation#Controlling minor ticks and labels

import string# Used for labelling panels in plots
import matplotlib.transforms as mtransforms# Used for labelling panels in plots

# Define some useful functions:
#Convenient for saving
def save(save_str, save_dict):
    with open(save_str + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(save_dict, json_file, ensure_ascii=False)

#Represent a list of numbers into a string:
def str_2(list_to_print, dp=6):
    str_to_print = '[' + str(round(list_to_print[0], dp))
    for num in list_to_print[1:]:
        str_to_print = str_to_print + ', ' + str(round(num, dp))
    str_to_print = str_to_print + ']'
    return str_to_print

# Improve tick marks (obtain at least three within the limits shown):
def fix_y_ticks(ax, remove_minor_ticks=True):
    if remove_minor_ticks:
        ax.yaxis.set_minor_locator(NullLocator())
    y_lim = ax.get_ylim()
    y_ticks = ax.get_yticks()
    y_ticks = y_ticks[(y_ticks >= y_lim[0]) * (y_ticks <= y_lim[1])]
    # if (len(y_ticks) >= 3):
    #     return
    y_scale = ax.get_yscale()
    if (y_scale == 'log'):
        y_ticks_0 = 10**np.arange(np.floor(np.log10(y_lim[0])), np.ceil(np.log10(y_lim[1])) + 0.5)
        y_ticks = y_ticks_0
        y_ticks_1 = np.concatenate([y_ticks_0, 0.3*y_ticks_0])
        y_ticks_2 = np.concatenate([y_ticks, 0.2*y_ticks, 0.5*y_ticks])
        y_ticks_list = []
        for n in [100, 50, 30, 15, 12, 9, 6, 3, 2]:
            y_ticks_n = np.concatenate([np.arange(0, np.floor(np.log10(y_lim[0])) - 0.5, -n), np.arange(0, np.ceil(np.log10(y_lim[1])) + 0.5, n)])
            y_ticks_n = np.unique(y_ticks_n)
            y_ticks_n = 10**y_ticks_n
            y_ticks_list += [y_ticks_n]
        y_ticks_list += [y_ticks_0, y_ticks_1, y_ticks_2]
        y_ticks_list = [y_ticks[(y_ticks >= y_lim[0]) * (y_ticks <= y_lim[1])] for y_ticks in y_ticks_list]
        for y_ticks in y_ticks_list:
            if (len(y_ticks) >= 3):
                ax.set_yticks(y_ticks)
                return
    else:
        y_upper_round = 10**np.ceil(np.log10(max(np.abs(y_lim))))
        y_tick_incr_scale_factor = y_upper_round
        attempt = 0
        while (attempt < 10**4):
            attempt += 1
            y_tick_incr_scale_factor *= 0.1
            y_ticks_list = []
            for y_incr_digit in [5, 2, 1]:
                y_incr = y_incr_digit*y_tick_incr_scale_factor
                y_ticks = np.arange(-y_upper_round, y_upper_round + 0.5*y_incr, y_incr)
                y_ticks = y_ticks[(y_ticks >= y_lim[0]) * (y_ticks <= y_lim[1])]
                y_ticks_list = y_ticks_list + [y_ticks]
            for y_ticks in y_ticks_list:
                if (len(y_ticks) >= 3):
                    ax.set_yticks(y_ticks)
                    return
    print('Did not achieve desired y-tick labels')

def fix_x_ticks(ax, remove_minor_ticks=True):
    if remove_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
    x_lim = ax.get_xlim()
    x_ticks = ax.get_xticks()
    x_ticks = x_ticks[(x_ticks >= x_lim[0]) * (x_ticks <= x_lim[1])]
    # if (len(x_ticks) >= 3):
    #     return
    x_scale = ax.get_xscale()
    if (x_scale == 'log'):
        x_ticks_0 = 10**np.arange(np.floor(np.log10(x_lim[0])), np.ceil(np.log10(x_lim[1])) + 0.5)
        x_ticks = x_ticks_0
        x_ticks_1 = np.concatenate([x_ticks_0, 0.3*x_ticks_0])
        x_ticks_2 = np.concatenate([x_ticks, 0.2*x_ticks, 0.5*x_ticks])
        x_ticks_list = []
        for n in [100, 50, 30, 15, 12, 9, 6, 3, 2]:
            x_ticks_n = np.concatenate([np.arange(0, np.floor(np.log10(x_lim[0])) - 0.5, -n), np.arange(0, np.ceil(np.log10(x_lim[1])) + 0.5, n)])
            x_ticks_n = np.unique(x_ticks_n)
            x_ticks_n = 10**x_ticks_n
            x_ticks_list += [x_ticks_n]
        x_ticks_list += [x_ticks_0, x_ticks_1, x_ticks_2]
        x_ticks_list = [x_ticks[(x_ticks >= x_lim[0]) * (x_ticks <= x_lim[1])] for x_ticks in x_ticks_list]
        for x_ticks in x_ticks_list:
            if (len(x_ticks) >= 3):
                ax.set_xticks(x_ticks)
                return
    else:
        x_upper_round = 10**np.ceil(np.log10(max(np.abs(x_lim))))
        x_tick_incr_scale_factor = x_upper_round
        attempt = 0
        while (attempt < 10**4):
            attempt += 1
            x_tick_incr_scale_factor *= 0.1
            x_ticks_list = []
            for x_incr_digit in [5, 2, 1]:
                x_incr = x_incr_digit*x_tick_incr_scale_factor
                x_ticks = np.arange(-x_upper_round, x_upper_round + 0.5*x_incr, x_incr)
                x_ticks = x_ticks[(x_ticks >= x_lim[0]) * (x_ticks <= x_lim[1])]
                x_ticks_list = x_ticks_list + [x_ticks]
            for x_ticks in x_ticks_list:
                if (len(x_ticks) >= 3):
                    ax.set_xticks(x_ticks)
                    return
    print('Did not achieve desired x-tick labels')
    
def beautify_distribution_name(distribution):
    if (distribution=='powerlaw'):
        distribution_new='Powerlaw'
    elif (distribution=='lognorm'):
        distribution_new='Lognormal'
    elif (distribution=='expon'):
        distribution_new='Exponential'
    elif (distribution=='truncnorm'):
        #distribution_new='truncated\nGaussian'
        distribution_new='Gaussian'
    elif (distribution=='uniform'):
        distribution_new='Uniform'
    return distribution_new
    
def beautify_stat_name(stat):
    stat_new = stat
    if (stat=='ks_stat'):
        stat_new='KS distance'
    elif (stat=='mean_val'):
        stat_new='Mean'
    elif (stat=='variance'):
        stat_new='Variance'
    elif (stat=='coef_var'):
        stat_new='Coef. of variation'
    elif (stat=='max_val'):
        stat_new='Maximum'
    elif (stat=='log_skew'):
        stat_new='Log-skewness'
    elif (stat=='kurtosis'):
        stat_new='Kurtosis'
    elif (stat=='skew'):
        stat_new='Skewness'
    elif (stat=='log_kurt'):
        stat_new='Log-kurtosis'
    return stat_new
    
def beautify_stat_name_concise(stat):
    stat_new = stat
    if (stat=='ks_stat'):
        stat_new='KS dist.'
    elif (stat=='max_val'):
        stat_new='Max.'
    elif (stat=='log_kurt'):
        stat_new='Log-kurt.'
    return stat_new
    
def beautify_method_name(method):
    method_new = method
    if (method=='constrained'):
        method_new='Constrained'
    elif (method=='typical'):
        method_new='Typical'
    elif (method=='original'):
        method_new='Original'
    elif (method=='true'):
        method_new='True'
    elif (method=='bootstrap'):
        method_new='Bootstrapped'
    return method_new


# Define some statistics:
# Some statistics
def mean_val(val_seq):
    return np.mean(val_seq);
def second_mom(val_seq):#Second moment
    return np.mean([val**2 for val in val_seq])
def third_mom(val_seq):#Third moment
    return np.mean([val**3 for val in val_seq])
def fourth_mom(val_seq):#Fourth moment
    return np.mean([val**4 for val in val_seq])
def max_val(val_seq):
    return np.max(val_seq);
def variance(val_seq):
    return np.var(val_seq);
def coef_var(val_seq):#Coefficient of variation (using population standard deviation)
    std_dev = np.sqrt(np.var(val_seq))
    mean_val = np.mean(val_seq)
    return (std_dev/mean_val);
def dispersion(val_seq):#Dispersion: ratio of variance to mean (using population standard deviation)
    return np.var(val_seq)/np.mean(val_seq);
def evi_mom(val_seq):#Moment estimator of extreme value index, based on Voitalov et al. (2020), "Scale-free networks well done"
    N = len(val_seq)
    val_seq = sorted(val_seq, reverse=True)
    val_min = val_seq[-1]
    val_max = val_seq[0]
    if (val_min == val_max):
        mom_est = 0
    else:
        log_val_rat_seq = [np.log(val/val_min) for val in val_seq]
        hill_est = np.sum(log_val_rat_seq)/(N - 1)#Hill estimator with k = N - 1
        log_squ_val_rat_seq = [val**2 for val in log_val_rat_seq]
        hill2_est = np.sum(log_squ_val_rat_seq)/(N - 1)#Hill, 2 function with k = N - 1
        if (1 == hill_est**2/hill2_est):
            mom_est = 0
        else:
            mom_est = hill_est + 1 - 0.5/(1 - hill_est**2/hill2_est)
    return mom_est
def evi_smooth(val_seq):#Smooth Hill estimator, r = 2, based on Voitalov et al. (2020), "Scale-free networks well done"
    N = len(val_seq)
    if (N <= 2):
        return np.nan
    def hill_est(sorted_val_seq):#Hill estimator with k = (len(sorted_val_seq) - 1)
        val_min = sorted_val_seq[-1]
        N = len(sorted_val_seq)
        log_val_rat_seq = [np.log(val/val_min) for val in sorted_val_seq]
        hill_est = np.sum(log_val_rat_seq)/(N - 1)
        return hill_est
    k = int(np.floor((N - 1)/2))#Require 2*k + 1 <= N
    val_seq = sorted(val_seq, reverse=True)
    hill_est_list = [hill_est(val_seq[0:j]) for j in range(k + 2, 2*k + 2)]#val_seq[0:(k' + 2)] has length k' + 2, so hill_est(val_seq[0:j]) will be the Hill estimate with k = k' + 1. val_seq[0:(2*k' + 1)] has length 2*k' + 1, so hill_est(val_seq[0:(2*k' + 1)]) will be the Hill estimate with k = 2*k'.
    smooth_hill_est = np.mean(hill_est_list)#Smooth Hill estimator with k = N/2, r = 2
    return smooth_hill_est
def geom_mean(val_seq):#Geometric mean
    return gmean(val_seq)
def harm_mean(val_seq):#Harmonic mean
    return hmean(val_seq)
def rang(val_seq):#Range
    return max(val_seq) - min(val_seq)
def log_rang(val_seq):#Range of log of values
    return np.log(max(val_seq)) - np.log(min(val_seq))
def iq_rang(val_seq):#Interquartile range
    return iqr(val_seq)
def log_iq_rang(val_seq):#Interquartile range of log of values
    return iqr(np.log(val_seq))
def log_mean(val_seq):#Mean of log of values
    return mean_val(np.log(val_seq))
def log_var(val_seq):#Variance of log of values
    return variance(np.log(val_seq))
def log_skew(val_seq):#Skewness of log of values
    return skew(np.log(val_seq))
def log_kurt(val_seq):#Kurtosis of log of values
    return kurtosis(np.log(val_seq))
def log_kurt(val_seq):#Kurtosis of log of values
    return kurtosis(np.log(val_seq))
def jb_stat(val_seq):
    skew_val = skew(val_seq, bias=False)#Unbiased skewness
    kurt_val = kurtosis(val_seq, fisher=False, bias=False)#Pearson's kurtosis instead of Fisher's excess kurtosis
    return (len(val_seq)/6)*(skew_val**2 + (kurt_val - 3)**2/4)
def log_jb_stat(val_seq):#Jarque-Bera test statistic for log of values
    return jb_stat(np.log(val_seq))
def dap_stat(val_seq):
    n = len(val_seq)
    skew_val = skew(val_seq, bias=False)
    kurt_val = kurtosis(val_seq, fisher=False, bias=False)#Pearson's kurtosis instead of Fisher's excess kurtosis
    skew_component = skew_val**2*(n/6)
    kurt_component = (kurt_val - 3)**2*(n/24)
    return skew_component + kurt_component
def log_dap_stat(val_seq):#D’Agostino-Pearson omnibus test statistic for log of values
    return dap_stat(np.log(val_seq))
    
    
# Fitting and calculating model selection criteria:

# Code for checking whether fit to doubly truncated normal is possible

def calc_H(theta1, t1):
    if not (theta1 == 0):
        H = 1/np.tanh(theta1) - 1/theta1 - t1
    else:
        H = 0
    return H

def calc_theta1_hat(t1):
    if (t1 == 0):
        theta1_hat = 0
    else:
        theta10 = 3*t1#Guess based on linearisation at theta1 == 0
        sol = root(lambda theta1 : calc_H(theta1, t1), theta10, jac=calc_dHdtheta1, method='hybr')#method='hybr' is default
        theta1_hat = sol.x[0]
    return theta1_hat

def return_cdf_func(param_hat, model='powerlaw', lower_cutoff_hat=1):
    
    if (model == 'powerlaw'):
        lambda_hat = param_hat[0]
        cdf_fun = lambda seq : expon.cdf(np.log(seq), loc=np.log(lower_cutoff_hat), scale=1/lambda_hat)
    elif (model == 'lognorm'):#Truncated lognormal distribution
        if (not np.isinf(param_hat[1])):#Could fit lognormal using maximum likelihood
            mu_hat, sigma_hat = param_hat[0], param_hat[1]
            cdf_fun = lambda seq : truncnorm.cdf((np.log(seq) - mu_hat)/sigma_hat, (np.log(lower_cutoff_hat) - mu_hat)/sigma_hat, (np.inf - mu_hat)/sigma_hat)
        else:#Could not fit lognormal using maximum likelihood: consider powerlaw instead
            lambda_hat = param_hat[0]
            cdf_fun = lambda seq : expon.cdf(np.log(seq), loc=np.log(lower_cutoff_hat), scale=1/lambda_hat)
    elif (model == 'expon'):#Exponential distribution
        lambda_hat = param_hat[0]
        cdf_fun = lambda seq : expon.cdf(seq, loc=lower_cutoff_hat, scale=1/lambda_hat)
    elif (model == 'truncnorm'):#Truncated Gaussian distribution
        if (not np.isinf(param_hat[1])):#Could fit truncated normal using maximum likelihood
            mu_hat, sigma_hat = param_hat[0], param_hat[1]
            cdf_fun = lambda seq : truncnorm.cdf((seq - mu_hat)/sigma_hat, (lower_cutoff_hat - mu_hat)/sigma_hat, (np.inf - mu_hat)/sigma_hat)
        else:#Could not fit truncated normal using maximum likelihood: consider exponential instead
            lambda_hat = param_hat[0]
            cdf_fun = lambda seq : expon.cdf(seq, loc=lower_cutoff_hat, scale=1/lambda_hat)
    elif (model == 'uniform'):#Uniform distribution
        b_hat = param_hat[0]
        a = lower_cutoff_hat
        cdf_fun = lambda seq : uniform.cdf(seq, loc=a, scale=(b_hat - a))
    else:
        raise Exception('model=' + str(model) + ' not recognised.')
    return cdf_fun

# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
def ks_stat(exp_cdf_with_rep, emp_cdf_with_rep, ks_method='sup'):#KS statistic
    if (ks_method == 'sup'):
        emp_cdf_with_rep_2 = np.concatenate(([0], emp_cdf_with_rep[:-1]))# Limit of empirical cdf as we approach from left
        emp_cdf_with_rep = np.concatenate((emp_cdf_with_rep, emp_cdf_with_rep_2))
        exp_cdf_with_rep = np.concatenate((exp_cdf_with_rep, exp_cdf_with_rep))
    diff_array = np.subtract(emp_cdf_with_rep, exp_cdf_with_rep)
    abs_diff_array = np.absolute(diff_array)
    ind_max_ks = np.argmax(abs_diff_array)
    ks = abs_diff_array[ind_max_ks]
    # return ks, val, abs_diff_array, eval_points
    return ks

# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
def kuiper_stat(exp_cdf_with_rep, emp_cdf_with_rep, kuiper_method='sup'):#KS statistic
    if (kuiper_method == 'sup'):
        emp_cdf_with_rep_2 = np.concatenate(([0], emp_cdf_with_rep[:-1]))# Limit of empirical cdf as we approach from left
        emp_cdf_with_rep = np.concatenate((emp_cdf_with_rep, emp_cdf_with_rep_2))
        exp_cdf_with_rep = np.concatenate((exp_cdf_with_rep, exp_cdf_with_rep))
    diff_array = np.subtract(emp_cdf_with_rep, exp_cdf_with_rep)
    kuiper = np.max(diff_array) + np.max(-diff_array)
    return kuiper

# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
def ad_stat(exp_cdf_with_rep, emp_cdf_with_rep=None):#Anderson-Darling statistic: https://encyclopediaofmath.org/wiki/Anderson-Darling_statistic
    eps = 1e-12
    exp_cdf_with_rep = np.clip(exp_cdf_with_rep, eps, 1 - eps)
    n = len(exp_cdf_with_rep)
    ind_array = np.arange(1, n + 1)
    summand_array = (2*ind_array - 1)*(np.log(exp_cdf_with_rep) + np.log(1 - exp_cdf_with_rep[::-1]))#Written this way to reduce numerical errors
    ad = -n - np.sum(summand_array)/n
    return ad

# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
def cvm_stat(exp_cdf_with_rep, emp_cdf_with_rep=None):#Cramer-von Mises statistic: https://encyclopediaofmath.org/wiki/Cram%C3%A9r-von_Mises_test
    n = len(exp_cdf_with_rep)
    ind_array = np.arange(1, n + 1)
    summand_array = (exp_cdf_with_rep - (2*ind_array - 1)/(2*n))**2
    cvm = np.sum(summand_array) + 1/(12*n)
    return cvm
    
# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
# Hybrid
def zk_stat(exp_cdf_with_rep, emp_cdf_with_rep=None):#Zhang statistic Z_K from Eq. (3.1) of "Powerful goodness-of-fit tests based on the likelihood ratio" by Jin Zhang (2002)
#Hybrid:
    eps = 1e-12
    exp_cdf_with_rep = np.clip(exp_cdf_with_rep, eps, 1 - eps)
    n = len(exp_cdf_with_rep)
    ind_array = np.arange(1, n + 1)
    cand_max_array = (ind_array - 1/2)*np.log((ind_array - 1/2)/(n*exp_cdf_with_rep)) + (n - ind_array + 1/2)*np.log((n - ind_array + 1/2)/(n*(1 - exp_cdf_with_rep)))
    zk = np.max(cand_max_array)
    return zk
    
# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
def za_stat(exp_cdf_with_rep, emp_cdf_with_rep=None):#Zhang statistic Z_K from Eq. (3.1) of "Powerful goodness-of-fit tests based on the likelihood ratio" by Jin Zhang (2002)
# Tail-sensitive:
    eps = 1e-12
    exp_cdf_with_rep = np.clip(exp_cdf_with_rep, eps, 1 - eps)
    n = len(exp_cdf_with_rep)
    ind_array = np.arange(1, n + 1)
    summand_array = np.log(exp_cdf_with_rep)/(n - ind_array + 1/2) + np.log(1 - exp_cdf_with_rep)/(ind_array - 1/2)
    za = -np.sum(summand_array)
    return za
    
# In the following, exp_cdf_with_rep (emp_cdf_with_rep) should be values of the theoretical (empirical) cdf in increasing order, with repetitions as necessary for repeated values of the considered variable
def zc_stat(exp_cdf_with_rep, emp_cdf_with_rep=None):#Zhang statistic Z_K from Eq. (3.1) of "Powerful goodness-of-fit tests based on the likelihood ratio" by Jin Zhang (2002)
# Centre-sensitive:
    eps = 1e-12
    exp_cdf_with_rep = np.clip(exp_cdf_with_rep, eps, 1 - eps)
    n = len(exp_cdf_with_rep)
    ind_array = np.arange(1, n + 1)
    summand_array = (np.log((exp_cdf_with_rep**-1 - 1)/((n - 1/2)/(ind_array - 3/4) - 1)))**2
    zc = np.sum(summand_array)
    return zc

# Find maximum likelihood fit to powerlaw, lognorm, expon, truncnorm or uniform model
def fit_model(val_seq, model='powerlaw', lower_cutoff_hat=None, param0=None):
    if (len(val_seq) == 0):
        lp_seq = np.array([])
        if (model == 'powerlaw') or (model == 'expon') or (model == 'uniform'):#One parameter
            param_hat = [np.nan]
        if (model == 'truncnorm') or (model == 'lognorm'):#Two parameters
            param_hat = [np.nan, np.nan]
        if (model == 'powerlaw_t') or (model == 'expon_t') or (model == 'uniform_t'):#Three parameters
            param_hat = [np.nan, np.nan, np.nan]
        if (model == 'truncnorm_t') or (model == 'lognorm_t'):#Four parameters
            param_hat = [np.nan, np.nan, np.nan, np.nan]
        return param_hat, lp_seq
    safety_factor_bounds = 1 + 10**-9
    if (lower_cutoff_hat == None):
        lower_cutoff_hat = int(np.floor(min(val_seq)))
    if (min(val_seq) < lower_cutoff_hat):
        raise Exception('Minimum of val_seq is smaller than lower_cutoff_hat.')
    if (model == 'powerlaw') or (model == 'lognorm'):
        if (model == 'powerlaw'):# Fit to power-law distribution with support [lower_cutoff, np.inf)
            model = 'expon'
        elif (model == 'lognorm'):# Fit to truncated lognormal with support [lower_cutoff, np.inf)
            model = 'truncnorm'
        param_hat, lp_seq = fit_model(np.log(val_seq), model=model, lower_cutoff_hat=np.log(lower_cutoff_hat), param0=param0)
        lp_seq = lp_seq - np.log(val_seq)
    elif (model == 'powerlaw_t') or (model == 'lognorm_t'):#Work in log-transformed space
        if (model == 'powerlaw_t'):# Fit to tightly bounded power-law distribution with support [lower_cutoff, np.inf)
            model = 'expon_t'
        elif (model == 'lognorm_t'):# Fit to tightly bounded truncated lognormal with support [lower_cutoff, np.inf)
            model = 'truncnorm_t'
        if not (param0==None):
            a0, b0 = np.log(param0[-2:])
            param0 = param0[:-2] + [a0, b0]
        param_hat, lp_seq = fit_model(np.log(val_seq), model=model, param0=param0)
        lp_seq = lp_seq - np.log(val_seq)
        a_hat, b_hat = np.exp(param_hat[-2:])
        param_hat = param_hat[:-2] + [a_hat, b_hat]
    else:
        if (model[-2:] == '_t'):#Using a tightly bounded model
            a_hat = min(val_seq)/safety_factor_bounds#Slightly decrease lower bound so that numerical errors do not move obervations outside of estimated support
            b_hat = max(val_seq)*safety_factor_bounds#Slightly increase upper bound so that numerical errors do not move obervations outside of estimated support
            if (model == 'expon_t'):# Fit to exponential distribution with support [lower_cutoff_hat, upper_cutoff_hat) by minimising negative log-likelihood per observation
                if (param0 == None):
                    lambda0 = 1/np.mean(np.array(val_seq) - a_hat)
                else:
                    lambda0 = param0[0]
                def func(p, val_seq, a_hat, b_hat):#Negative log-likelihood per observation
                    lambda_hat = p
                    return -np.mean(expon.logpdf(val_seq, a_hat, scale=1/lambda_hat)) + np.log(expon.cdf(b_hat, a_hat, scale=1/lambda_hat))
                opt_result = minimize(func, [lambda0], args=(val_seq, a_hat, b_hat), bounds=[(None,None)])
                param_hat = opt_result.x
                nllpo = opt_result.fun
                lambda_hat = param_hat[0]
                lp_seq = expon.logpdf(val_seq, a_hat, scale=1/lambda_hat) - np.log(expon.cdf(b_hat, a_hat, scale=1/lambda_hat))
            elif (model == 'truncnorm_t'):# Fit to truncated normal with support [lower_cutoff_hat, upper_cutoff_hat) by minimising negative log-likelihood per observationif (param0 == None):
                val_seq = np.array(val_seq)
                t1 = np.mean(val_seq - a_hat)
                t2 = np.mean((val_seq - a_hat)**2)
                u = t2/t1**2 - 1
                theta1_hat = calc_theta1_hat(t1)
                u_upper = 1/t1**2 - 2/(theta1_hat*t1) - 1
                if (u < u_upper):#'Maximum likelihood parameters mu, sigma ARE finite'#See Hedge and Dahiya (1989) Estimation of the parameters in a truncated normal distribution
                    if (param0 == None):
                        # mu0 = np.mean(val_seq)
                        # sigma0 = np.std(val_seq)
                        mu0 = a_hat
                        sigma0 = np.std(val_seq - mu0)
                    else:
                        mu0, sigma0 = param0[0], param0[1]
                    def func(p, val_seq, a_hat, b_hat):#Negative log-likelihood per observation
                        mu_hat, sigma_hat = p
                        return -np.mean(truncnorm.logpdf(val_seq, (a_hat - mu_hat)/sigma_hat, (b_hat - mu_hat)/sigma_hat, loc=mu_hat, scale=sigma_hat))
                    opt_result = minimize(func, [mu0, sigma0], args=(val_seq, a_hat, b_hat), bounds=[(None,None),(10**-12,None)])
                    param_hat = opt_result.x
                    nllpo = opt_result.fun
                    mu_hat, sigma_hat = param_hat[0], param_hat[1]
                    lp_seq = truncnorm.logpdf(val_seq, (a_hat - mu_hat)/sigma_hat, (b_hat - mu_hat)/sigma_hat, loc=mu_hat, scale=sigma_hat)
                else:#'Maximum likelihood parameters mu, sigma ARE NOT finite' - fit an exponential distribution instead
                    if (param0 == None):
                        lambda0 = 1/np.mean(np.array(val_seq) - a_hat)
                    else:
                        mu0, sigma0 = param0[0], param0[1]
                        lambda0 = -(mu - a_hat)/sigma0**2
                    param_hat, lp_seq = fit_model(val_seq, model='expon_t', param0=[lambda0])
                    lambda_hat = param_hat[0]
                    param_hat = [lambda_hat, np.inf]
            elif (model == 'uniform_t'):# Fit to uniform distribution
                # For consistency with other fitting methods, perhaps I should consider numerically minimising the negative log-likelihood
                param_hat = [b_hat]#Later will " + [a_hat, b_hat]", which leads to redundancy in param_hat
                nllpo = np.log(b_hat - a_hat)
                lp_seq = np.array([-nllpo for val in val_seq])
            else:
                raise Exception('model=' + str(model) + ' not recognised.')
            param_hat = list(param_hat) + [a_hat, b_hat]
        else:#Not using a tightly bounded model
            if (model == 'expon'):# Fit to exponential distribution with support [lower_cutoff, np.inf) by minimising negative log-likelihood per observation
                if (param0 == None):
                    lambda0 = 1/np.mean(np.array(val_seq) - lower_cutoff_hat)
                else:
                    lambda0 = param0[0]
                def func(p, val_seq, lower_cutoff_hat):#Negative log-likelihood per observation
                    lambda_hat = p
                    return -np.mean(expon.logpdf(val_seq, lower_cutoff_hat, scale=1/lambda_hat))
                opt_result = minimize(func, [lambda0], args=(val_seq, lower_cutoff_hat), bounds=[(10**-12,None)])
                param_hat = opt_result.x
                nllpo = opt_result.fun
                lambda_hat = param_hat[0]
                lp_seq = expon.logpdf(val_seq, lower_cutoff_hat, scale=1/lambda_hat)
            elif (model == 'truncnorm'):# Fit to truncated normal with support [lower_cutoff, np.inf) by minimising negative log-likelihood per observationif (param0 == None):
                val_seq = np.array(val_seq)
                t1 = np.mean(val_seq - lower_cutoff_hat)
                t2 = np.mean((val_seq - lower_cutoff_hat)**2)
                u = t2/t1**2 - 1
                if (u > 0) and (u < 1):#'Maximum likelihood parameters mu, sigma ARE finite'#See Hedge and Dahiya (1989) Estimation of the parameters in a truncated normal distribution
                    if (param0 == None):
                        # mu0 = np.mean(val_seq)
                        # sigma0 = np.std(val_seq)
                        mu0 = lower_cutoff_hat
                        sigma0 = np.std(val_seq - mu0)
                    else:
                        mu0, sigma0 = param0[0], param0[1]
                    def func(p, val_seq, lower_cutoff_hat):#Negative log-likelihood per observation
                        mu_hat, sigma_hat = p
                        return -np.mean(truncnorm.logpdf(val_seq, (lower_cutoff_hat - mu_hat)/sigma_hat, (np.inf - mu_hat)/sigma_hat, loc=mu_hat, scale=sigma_hat))
                    opt_result = minimize(func, [mu0, sigma0], args=(val_seq, lower_cutoff_hat), bounds=[(None,None),(10**-12,None)])
                    param_hat = opt_result.x
                    nllpo = opt_result.fun
                    mu_hat, sigma_hat = param_hat[0], param_hat[1]
                    lp_seq = truncnorm.logpdf(val_seq, (lower_cutoff_hat - mu_hat)/sigma_hat, np.inf, loc=mu_hat, scale=sigma_hat)
                else:#'Maximum likelihood parameters mu, sigma ARE NOT finite' - fit an exponential distribution instead
                    if (param0 == None):
                        lambda0 = 1/np.mean(np.array(val_seq) - lower_cutoff_hat)
                    else:
                        mu0, sigma0 = param0[0], param0[1]
                        lambda0 = -(mu - lower_cutoff_hat)/sigma0**2
                    param_hat, lp_seq = fit_model(val_seq, model='expon', lower_cutoff_hat=lower_cutoff_hat, param0=[lambda0])
                    lambda_hat = param_hat[0]
                    param_hat = [lambda_hat, np.inf]
            elif (model == 'uniform'):# Fit to uniform distribution
                # For consistency with other fitting methods, perhaps I should consider numerically minimising the negative log-likelihood
                a = lower_cutoff_hat
                b_hat = max(val_seq)
                param_hat = [b_hat]
                nllpo = np.log(b_hat - a)
                lp_seq = np.array([-nllpo for val in val_seq])
            else:
                raise Exception('model=' + str(model) + ' not recognised.')
    param_hat = list(param_hat)
    return param_hat, lp_seq

# Code for calculating empirical cdf:
def ecdf_2(val_seq, method='max'):
    if (method == 'min'):
        a = 1
    elif (method == 'max'):#The standard definition
        a = 0
    elif (method == 'average'):
        a = 0.5
    else:
        raise Exception('method=' + str(method) + ' not recognised.')
    val_seq = np.sort(val_seq)
    n = len(val_seq)
    val_seq_unique, unique_indices = np.unique(val_seq, return_index=True)
    emp_cdf = (rankdata(val_seq, method=method) - a)/n
    emp_cdf = list(emp_cdf[unique_indices])
    val_seq_unique = list(val_seq_unique)
    return val_seq_unique, emp_cdf

#Code for calculating KS-distance:
def calc_ks(val_seq, model='powerlaw', lower_cutoff_hat=None, param_hat=None, ks_method='sup'):
    
    if (len(val_seq) == 0):#A consistent choice for the case when it is hard to calculate the KS distance from its definition (chosen large because, when the two distributions are the same, teh KS distance tends to decrease with the number of samples)
        return 1
    
    if (lower_cutoff_hat == None):
        lower_cutoff_hat = np.floor(min(val_seq))
    if (min(val_seq) < lower_cutoff_hat):
        raise Exception('Minimum of val_seq is smaller than lower_cutoff_hat.')
    
    trunc_val_seq = [val for val in val_seq if val >= lower_cutoff_hat]
    unique_trunc_val_seq = np.unique(trunc_val_seq)
        
    if (model == 'powerlaw'):
        if (param_hat == None):#Added 2025.06
            param_hat, lp_seq = fit_model(trunc_val_seq, lower_cutoff_hat=lower_cutoff_hat, model='powerlaw')
        lambda_hat = param_hat[0]
        cdf_fun = lambda seq : expon.cdf(np.log(seq), loc=np.log(lower_cutoff_hat), scale=1/lambda_hat)
    elif (model == 'lognorm'):#Truncated lognormal distribution
        if (param_hat == None):#Added 2025.06
            param_hat, lp_seq = fit_model(trunc_val_seq, lower_cutoff_hat=lower_cutoff_hat, model='lognorm')
        if (not np.isinf(param_hat[1])):#Could fit lognormal using maximum likelihood
            mu_hat, sigma_hat = param_hat[0], param_hat[1]
            if (np.median((np.log(unique_trunc_val_seq) - mu_hat)/sigma_hat) < 0):
                cdf_fun = lambda seq : truncnorm.cdf((np.log(seq) - mu_hat)/sigma_hat, (np.log(lower_cutoff_hat) - mu_hat)/sigma_hat, (np.inf - mu_hat)/sigma_hat)
            else:
                cdf_fun = lambda seq : 1 - truncnorm.cdf(-(np.log(seq) - mu_hat)/sigma_hat, -(np.inf - mu_hat)/sigma_hat, -(np.log(lower_cutoff_hat) - mu_hat)/sigma_hat)
        else:#Could not fit lognormal using maximum likelihood: consider powerlaw instead
            lambda_hat = param_hat[0]
            cdf_fun = lambda seq : expon.cdf(np.log(seq), loc=np.log(lower_cutoff_hat), scale=1/lambda_hat)
    elif (model == 'expon'):#Exponential distribution
        if (param_hat == None):
            param_hat, lp_seq = fit_model(trunc_val_seq, lower_cutoff_hat=lower_cutoff_hat, model='expon')
        lambda_hat = param_hat[0]
        cdf_fun = lambda seq : expon.cdf(seq, loc=lower_cutoff_hat, scale=1/lambda_hat)
    elif (model == 'truncnorm'):#Truncated Gaussian distribution
        if (param_hat == None):
            param_hat, lp_seq = fit_model(trunc_val_seq, lower_cutoff_hat=lower_cutoff_hat, model='truncnorm')
        if (not np.isinf(param_hat[1])):#Could fit truncated normal using maximum likelihood
            mu_hat, sigma_hat = param_hat[0], param_hat[1]
            if (np.median((unique_trunc_val_seq - mu_hat)/sigma_hat) < 0):
                cdf_fun = lambda seq : truncnorm.cdf((seq - mu_hat)/sigma_hat, (lower_cutoff_hat - mu_hat)/sigma_hat, (np.inf - mu_hat)/sigma_hat)
            else:
                cdf_fun = lambda seq : 1 - truncnorm.cdf(-(seq - mu_hat)/sigma_hat, -(np.inf - mu_hat)/sigma_hat, -(lower_cutoff_hat - mu_hat)/sigma_hat)
        else:#Could not fit truncated normal using maximum likelihood: consider exponential instead
            lambda_hat = param_hat[0]
            cdf_fun = lambda seq : expon.cdf(seq, loc=lower_cutoff_hat, scale=1/lambda_hat)
    elif (model == 'uniform'):#Uniform distribution
        if (param_hat == None):
            param_hat, lp_seq = fit_model(trunc_val_seq, lower_cutoff_hat=lower_cutoff_hat, model='uniform')
        b_hat = param_hat[0]
        a = lower_cutoff_hat
        cdf_fun = lambda seq : uniform.cdf(seq, loc=a, scale=(b_hat - a))
    else:
        raise Exception('model=' + str(model) + ' not recognised.')
    #if (not continuous):#Considering (or comparing with) discrete (integer) data#Updated to consider this case 2024.02.19
    #    trunc_val_seq = [np.ceil(val - 0.5) + 0.5 for val in trunc_val_seq]#Round everything up to the nearest non-integer half integer (n + 1/2, where n is an integer) 
    if (ks_method == 'sup'):
        eval_points, emp_cdf = ecdf_2(trunc_val_seq, method='max')
        emp_cdf_2 = [0] + emp_cdf[:-1]# Limit of empirical cdf as we approach from left
        emp_cdf = emp_cdf + emp_cdf_2
        exp_cdf = cdf_fun(eval_points)
        exp_cdf = list(exp_cdf)
        exp_cdf = exp_cdf + exp_cdf
        eval_points = eval_points + eval_points
    elif (ks_method == 'ave'):
        eval_points, emp_cdf = ecdf_2(trunc_val_seq, method='average')
        exp_cdf = cdf_fun(eval_points)
        exp_cdf = list(exp_cdf)
    else:
        raise Exception('ks_method=' + str(ks_method) + ' not recognised.')
    #emp_cdf_2 = [0] + emp_cdf[:-1]# Limit of empirical cdf as we approach from left#Commented 2024.02.19
    diff_array = np.subtract(emp_cdf, exp_cdf)
    #if continuous:#Not considering (or comparing with) discrete (integer) data#Updated to separate this case 2024.02.19
    #    #We do not need to worry about this for the case where continuous==False because then we only evaluate at n + 1/2, where n is an integer (for either integer or continuous processes, there is vanishing chance that a value will be observed at such a point)
    #    emp_cdf_2 = [0] + emp_cdf[:-1]# Limit of empirical cdf as we approach from left
    #    emp_cdf = emp_cdf + emp_cdf_2
    #    exp_cdf = exp_cdf + exp_cdf
    #diff_array = np.subtract(emp_cdf + emp_cdf_2, exp_cdf + exp_cdf)#Commented 2024.02.19
    abs_diff_array = np.absolute(diff_array)
    ks = max(abs_diff_array)
    return ks

# Fitting:
# 

# Find KS distance-minimising integer lower cut-off for maximum likelihood powerlaw, lognorm, expon, truncnorm or uniform model
def fit_lower_cutoff(val_seq, model='powerlaw', continuous=True, ks_method='sup'):
    if (model[-2:] == '_t'):
        lower_cutoff_hat_seq = list(np.unique(val_seq))
    else:
        #lower_cutoff_hat_seq = np.unique(np.floor(val_seq).tolist() + np.ceil(val_seq).tolist())#Commented 2024.02.23
        if continuous:#Added 2024.02.23
            lower_cutoff_hat_seq = list(np.unique(np.floor(val_seq)))
            lower_cutoff_hat_seq = [int(lower_cutoff_hat) for lower_cutoff_hat in lower_cutoff_hat_seq]
        else:#(not continuous): Considering discrete (integer) data
            lower_cutoff_hat_seq = [np.floor(lower_cutoff_hat + 0.5) - 0.5 for lower_cutoff_hat in val_seq]#Added 2024.02.23: Consider each (n + 1/2) just before a value
            #lower_cutoff_hat_seq = [lower_cutoff_hat - 0.5 for lower_cutoff_hat in lower_cutoff_hat_seq]#Commented 2024.02.21
            #lower_cutoff_hat_seq = [lower_cutoff_hat - 0.5 for lower_cutoff_hat in lower_cutoff_hat_seq] + [lower_cutoff_hat + 0.5 for lower_cutoff_hat in lower_cutoff_hat_seq]#Added 2024.02.21#Commented 2024.02.23
            lower_cutoff_hat_seq = list(np.unique(lower_cutoff_hat_seq))
    ks_seq = [np.nan for lower_cutoff in lower_cutoff_hat_seq]
    for i_lower_cutoff_hat in range(len(lower_cutoff_hat_seq)):
        lower_cutoff_hat = lower_cutoff_hat_seq[i_lower_cutoff_hat]
        val_seq_trunc = [val for val in val_seq if val >= lower_cutoff_hat]
        ks = calc_ks(val_seq_trunc, model=model, lower_cutoff_hat=lower_cutoff_hat, param_hat=None, ks_method=ks_method)
        ks_seq[i_lower_cutoff_hat] = ks
    i_lower_cutoff_hat = np.nanargmin(ks_seq)
    lower_cutoff_hat = lower_cutoff_hat_seq[i_lower_cutoff_hat]
    ks = ks_seq[i_lower_cutoff_hat]
    return lower_cutoff_hat, ks, lower_cutoff_hat_seq, ks_seq
    
    
# Generating surrogates:

# Helper functions for generating surrogates:

#Bootstrapping
def bootstrap(val_seq, N=0):
    N0 = len(val_seq)
    if (N == 0):
        N = N0
    return random.choices(val_seq, k=N)

#Code for generating time series from known parameters:
def gen_data(model, N=1, lower_cutoff=1, param=None):
    if (N == 0):
        return []
    if (model == 'powerlaw') or (model == 'lognorm'):
        if (model == 'powerlaw'):
            model = 'expon'
        elif (model == 'lognorm'):
            model = 'truncnorm'
        log_val_seq = gen_data(model, N=N, lower_cutoff=np.log(lower_cutoff), param=param)
        val_seq = np.exp(log_val_seq)
    elif (model == 'powerlaw_t') or (model == 'lognorm_t'):
        if (model == 'powerlaw_t'):
            model = 'expon_t'
        elif (model == 'lognorm_t'):
            model = 'truncnorm_t'
        if not (param==None):
            a, b = np.log(param[-2:])
            param = param[:-2] + [a, b]
        log_val_seq = gen_data(model, N=N, param=param)
        val_seq = np.exp(log_val_seq)
    else:
        if (model[-2:] == '_t'):#Using a tightly bounded model
            if (model == 'expon_t'):
                if (param == None):
                    lam = 1
                    a, b = 1, 9
                else:
                    lam = param[0]
                    a, b = param[-2:]
                #val_seq = expon.rvs(loc=lower_cutoff, scale=1/lam, size=N)
                # From https://math.stackexchange.com/questions/788285/generate-exponential-random-values-in-a-given-range
                val_seq = uniform.rvs(loc=0, scale=1, size=N)
                val_seq = -(1/lam)*np.log(np.exp(-lam*a) - val_seq*(np.exp(-lam*a) - np.exp(-lam*b)))
            elif (model == 'truncnorm_t'):
                if (param == None):
                    param = [0, 1, 1, 9]
                if (not np.isinf(param[1])):
                    mu, sigma = param[0], param[1]
                    a, b = param[-2:]
                    val_seq = truncnorm.rvs((a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma, size=N)
                else:
                    lam = param[0]
                    a, b = param[-2:]
                    #val_seq = expon.rvs(loc=lower_cutoff, scale=1/lam, size=N)
                    # From https://math.stackexchange.com/questions/788285/generate-exponential-random-values-in-a-given-range
                    val_seq = uniform.rvs(loc=0, scale=1, size=N)
                    val_seq = -(1/lam)*np.log(np.exp(-lam*a) - val_seq*(np.exp(-lam*a) - np.exp(-lam*b)))
            elif (model == 'uniform_t'):
                if (param == None):
                    param = [9, 1, 9]
                a, b = param[-2:]
                val_seq = uniform.rvs(loc=a, scale=(b - a), size=N)
            else:
                raise Exception('model=' + str(model) + ' not recognised.')
        else:#Not using a tightly bounded model
            if (model == 'expon'):
                if (param == None):
                    lam = 1
                else:
                    lam = param[0]
                val_seq = expon.rvs(loc=lower_cutoff, scale=1/lam, size=N)
            elif (model == 'truncnorm'):
                if (param == None):
                    param = [0, 1]
                if (not np.isinf(param[1])):
                    mu, sigma = param[0], param[1]
                    val_seq = truncnorm.rvs((lower_cutoff - mu)/sigma, (np.inf - mu)/sigma, loc=mu, scale=sigma, size=N)
                else:
                    lam = param[0]
                    val_seq = expon.rvs(loc=lower_cutoff, scale=1/lam, size=N)
            elif (model == 'uniform'):
                a = lower_cutoff
                if (param == None):
                    b = 2*lower_cutoff
                else:
                    b = param[0]
                val_seq = uniform.rvs(loc=a, scale=(b - a), size=N)
            else:
                raise Exception('model=' + str(model) + ' not recognised.')
    val_seq = list(val_seq)
    return val_seq

# Generate constrained exponential surrogate by merging two values at a time
def gen_const_exp_surr_merge_2(val_seq, num_trans=None, lower_cutoff_hat=None, upper_cutoff_hat=None, centr_cutoff_hat=None):
    if (lower_cutoff_hat == None):
        lower_cutoff_hat = np.floor(min(val_seq))
    if (centr_cutoff_hat == None):
        centr_cutoff_hat = lower_cutoff_hat
    if (upper_cutoff_hat == None):
        upper_cutoff_hat = np.inf
    
    N = len(val_seq)
    if (num_trans == None):
        num_trans = N*(int(np.ceil(np.log2(N*1024))))
    
    if (N <= 1):
        surr_val_seq = val_seq
        np.random.shuffle(surr_val_seq)
        return list(surr_val_seq)
    
    surr_val_seq = [val - lower_cutoff_hat for val in val_seq]
    indices = [ii for ii in range(N)]
    t = 0
    while (t < num_trans):
        t += 1
        from_to_indices = random.sample(indices, 2)
        i = from_to_indices[0]
        j = from_to_indices[1]
        x_i = surr_val_seq[i]
        x_j = surr_val_seq[j]
        x_i_candidate = np.random.uniform(0, min([x_i + x_j, upper_cutoff_hat - lower_cutoff_hat]))
        x_j_candidate = x_i + x_j - x_i_candidate
        if (not (min([x_i_candidate, x_j_candidate]) >= 0)):#Try again
            t -= 1
            continue
        if (not (max([x_i_candidate, x_j_candidate]) <= upper_cutoff_hat - lower_cutoff_hat)):#Candidate transition would lead to a value larger than estimated upper cutoff
            t -= 1
            continue
        if (not (sum([x_i_candidate > centr_cutoff_hat, x_j_candidate > centr_cutoff_hat]) == sum([x_i > centr_cutoff_hat, x_j > centr_cutoff_hat]))):#Candidate transition would change the count below and above the threshold centr_cutoff_hat
            t -= 1
            continue
        surr_val_seq[i] = x_i_candidate
        surr_val_seq[j] = x_j_candidate
    
    surr_val_seq = [val + lower_cutoff_hat for val in surr_val_seq]
    
    return surr_val_seq

# Generate constrained truncated normal surrogate by merging three values at a time via picking a point on the circle comprising the intersection of x + y + z = m_1 and x**2 + y**2 + z**2 = m_2
# Version 1, which appears (counterintuitively) faster than Version 2
def gen_const_t_n_surr_merge_3(val_seq, num_trans=None, lower_cutoff_hat=None, upper_cutoff_hat=None, centr_cutoff_hat=None):
    if (lower_cutoff_hat == None):
        lower_cutoff_hat = np.floor(min(val_seq))
    if (centr_cutoff_hat == None):
        centr_cutoff_hat = lower_cutoff_hat
    if (upper_cutoff_hat == None):
        upper_cutoff_hat = np.inf
    
    N = len(val_seq)
    if (num_trans == None):
        num_trans = N*(int(np.ceil(np.log2(N*1024))))
    
    if (N <= 2):
        surr_val_seq = val_seq
        np.random.shuffle(surr_val_seq)
        return list(surr_val_seq)
        
    surr_val_seq = [val - lower_cutoff_hat for val in val_seq]
    surr_lower_cutoff_hat = lower_cutoff_hat - lower_cutoff_hat#This should be zero
    indices = [ii for ii in range(N)]
    #U = np.array([[0, np.sqrt(2/3), 1/np.sqrt(3)], [-1/np.sqrt(2), -1/np.sqrt(6), 1/np.sqrt(3)], [1/np.sqrt(2), -1/np.sqrt(6), 1/np.sqrt(3)]])
    #V = np.array([[0, np.sqrt(2/3)], [-1/np.sqrt(2), -1/np.sqrt(6)], [1/np.sqrt(2), -1/np.sqrt(6)]])
    W = np.array([[0, np.sqrt(2/3)], [-1/np.sqrt(2), -1/np.sqrt(6)]])
    t = 0
    while (t < num_trans):
        t += 1
        from_to_indices = random.sample(indices, 3)
        i = from_to_indices[0]
        j = from_to_indices[1]
        k = from_to_indices[2]
        x_i = surr_val_seq[i]
        x_j = surr_val_seq[j]
        x_k = surr_val_seq[k]
        m_1 = x_i + x_j + x_k
        m_2 = x_i**2 + x_j**2 + x_k**2
        th = np.random.uniform(0, 2*np.pi)
        #x_ijk_candidate = np.sqrt(3*m_2 - m_1**2)/np.sqrt(3)*U.dot([np.cos(th), np.sin(th), 0]) + m_1/3
        #x_ijk_candidate = np.sqrt(3*m_2 - m_1**2)/np.sqrt(3)*V.dot([np.cos(th), np.sin(th)]) + m_1/3
        #x_i_candidate, x_j_candidate, x_k_candidate = x_ijk_candidate[0], x_ijk_candidate[1], x_ijk_candidate[2]
        x_ij_candidate = np.sqrt(3*m_2 - m_1**2)/np.sqrt(3)*W.dot([np.cos(th), np.sin(th)]) + m_1/3
        x_i_candidate, x_j_candidate = x_ij_candidate[0], x_ij_candidate[1]
        x_k_candidate = m_1 - x_i_candidate - x_j_candidate
        if (not (min([x_i_candidate, x_j_candidate, x_k_candidate]) >= 0)):#Try again
            t -= 1
            continue
        if (not (max([x_i_candidate, x_j_candidate, x_k_candidate]) <= upper_cutoff_hat - lower_cutoff_hat)):#Candidate transition would lead to a value larger than estimated upper cutoff
            t -= 1
            continue
        if (not (sum([x_i_candidate > centr_cutoff_hat, x_j_candidate > centr_cutoff_hat, x_k_candidate > centr_cutoff_hat]) == sum([x_i > centr_cutoff_hat, x_j > centr_cutoff_hat, x_k > centr_cutoff_hat]))):#Candidate transition would change the count below and above the threshold centr_cutoff_hat
            t -= 1
            continue
        
        surr_val_seq[i] = x_i_candidate
        surr_val_seq[j] = x_j_candidate
        surr_val_seq[k] = x_k_candidate
    
    surr_val_seq = [val + lower_cutoff_hat for val in surr_val_seq]
    
    return surr_val_seq

#Code for generating surrogates
def gen_surrogate(val_seq, model='powerlaw', num_trans=None, method='constrained', lower_cutoff_hat=None, upper_cutoff_hat=None, centr_cutoff_hat=None, param_hat=None):
    N = len(val_seq)
    if (num_trans == None):
        num_trans = N*(int(np.ceil(np.log2(N*1024))))
    if (N == 0):
        return []
    
    if (lower_cutoff_hat == None):
        lower_cutoff_hat = np.floor(min(val_seq))
    if (centr_cutoff_hat == None):
        centr_cutoff_hat = lower_cutoff_hat
    if (upper_cutoff_hat == None):
        upper_cutoff_hat = np.inf
    
    if (min(val_seq) < lower_cutoff_hat):
        raise Exception('Minimum of val_seq is smaller than lower_cutoff_hat.')
    
    
    if (method == 'typical'):
        if (param_hat == None):
            param_hat, lp_seq = fit_model(val_seq, model=model, lower_cutoff_hat=lower_cutoff_hat, param0=None)
        surr_val_seq = gen_data(model, N=N, lower_cutoff=lower_cutoff_hat, param=param_hat)
            
    elif (method == 'constrained'):
        if (model[-2:] == '_t'):#Using a tightly bounded model
            i = np.argmin(val_seq)#Picking first instance of minimum value - this will be fine if order of surr_val_seq does not matter
            j = np.argmax(val_seq)#Picking first instance of maximum value - this will be fine if order of surr_val_seq does not matter
            lower_cutoff_hat = val_seq[i]
            upper_cutoff_hat = val_seq[j]
            surr_val_i, surr_val_j = lower_cutoff_hat, upper_cutoff_hat
            if (i > j):#Ensure i <= j
                i, j = j, i
            if (np.random.random() < 0.5):
                surr_val_i, surr_val_j = surr_val_j, surr_val_i
            val_seq_2 = val_seq[:i] + val_seq[(i + 1):j] + val_seq[(j + 1):]
            if (model == 'uniform_t'):
                method = 'typical'
            surr_val_seq_2 = gen_surrogate(val_seq_2, model=model[:-2], num_trans=num_trans, method=method, lower_cutoff_hat=lower_cutoff_hat, param_hat=param_hat, upper_cutoff_hat=upper_cutoff_hat)
            surr_val_seq = surr_val_seq_2[:i] + [surr_val_i] + surr_val_seq_2[i:(j - 1)] + [surr_val_j] + surr_val_seq_2[(j - 1):]
        else:
            if (model == 'powerlaw'):# Generate constrained powerlaw surrogate by merging two values at a time after taking logarithm
                surr_val_seq = [np.log(val/lower_cutoff_hat) for val in val_seq]
                surr_val_seq = gen_const_exp_surr_merge_2(surr_val_seq, num_trans, lower_cutoff_hat=0, upper_cutoff_hat=np.log(upper_cutoff_hat/lower_cutoff_hat), centr_cutoff_hat=np.log(centr_cutoff_hat/lower_cutoff_hat))
                surr_val_seq = [np.exp(val)*lower_cutoff_hat for val in surr_val_seq]
            elif (model == 'lognorm'):# Generate constrained lognormal surrogate by merging three values at a time after taking logarithm
                surr_val_seq = [np.log(val/lower_cutoff_hat) for val in val_seq]
                surr_val_seq = gen_const_t_n_surr_merge_3(surr_val_seq, num_trans, lower_cutoff_hat=0, upper_cutoff_hat=np.log(upper_cutoff_hat/lower_cutoff_hat), centr_cutoff_hat=np.log(centr_cutoff_hat/lower_cutoff_hat))
                surr_val_seq = [np.exp(val)*lower_cutoff_hat for val in surr_val_seq]
            elif (model == 'expon'):# Generate constrained exponential surrogate by merging two values at a time
                surr_val_seq = gen_const_exp_surr_merge_2(val_seq, num_trans, lower_cutoff_hat=lower_cutoff_hat, upper_cutoff_hat = upper_cutoff_hat, centr_cutoff_hat=centr_cutoff_hat)
            elif (model == 'truncnorm'):# Generate constrained truncated normal surrogate by merging three values at a time via picking a point on the circle comprising the intersection of x + y + z = m_1 and x**2 + y**2 + z**2 = m_2
                surr_val_seq = gen_const_t_n_surr_merge_3(val_seq, num_trans, lower_cutoff_hat=lower_cutoff_hat, upper_cutoff_hat = upper_cutoff_hat, centr_cutoff_hat=centr_cutoff_hat)
            elif (model == 'uniform'):# Generate constrained uniform surrogate
                val_seq_L = [val for val in val_seq if val < centr_cutoff_hat]
                val_seq_R = [val for val in val_seq if val >= centr_cutoff_hat]
                N_L, N_R = len(val_seq_L), len(val_seq_R)
                a = lower_cutoff_hat
                b_hat = max(val_seq)
                surr_val_seq_L = list(uniform.rvs(loc=a, scale=(centr_cutoff_hat - a), size=N_L))
                surr_val_seq_R = list(uniform.rvs(loc=centr_cutoff_hat, scale=(b_hat - centr_cutoff_hat), size=N_R))
                [i_max_val] = np.random.choice(N_R, 1, replace=False)
                surr_val_seq_R[i_max_val] = b_hat
                surr_val_seq = surr_val_seq_L + surr_val_seq_R
                np.random.shuffle(surr_val_seq)
            # elif (model == 'uniform'):# Generate constrained uniform surrogate
            #     [i_max_val] = np.random.choice(N, 1, replace=False)
            #     a = lower_cutoff_hat
            #     b_hat = max(val_seq)
            #     surr_val_seq = uniform.rvs(loc=a, scale=(b_hat - a), size=N)
            #     surr_val_seq[i_max_val] = b_hat
            else:
                raise Exception('model=' + str(model) + ' with method=' + str(method) + ' not recognised.')
        
    else:
        raise Exception('Surrogate method=' + str(method) + ' not recognised.')
    
    surr_val_seq = list(surr_val_seq)
    return surr_val_seq
    


#Code for setting font and font size in plots - base_font_size < 8 is probably too small
# def set_main_fonts(base_font_size=9, tick_scale=0.85, label_scale=1.1, title_scale=1.15, legend_scale=1.0):
def set_main_fonts(base_font_size=9, tick_scale=0.9, label_scale=1.0, title_scale=1.0, legend_scale=1.0):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": base_font_size,
        "axes.labelsize": base_font_size*label_scale,
        "axes.titlesize": base_font_size*title_scale,
        "xtick.labelsize": base_font_size*tick_scale,
        "ytick.labelsize": base_font_size*tick_scale,
        "legend.fontsize": base_font_size*legend_scale,
    })

# def set_inset_fonts(ax_inset, scale=0.75):
def set_inset_fonts(ax_inset, scale=0.9):
    base_font_size = plt.rcParams["font.size"]
    ax_inset.tick_params(axis="both", which="major", labelsize=plt.rcParams["xtick.labelsize"]*scale)
    ax_inset.xaxis.label.set_fontsize(plt.rcParams["axes.labelsize"]*scale)
    ax_inset.yaxis.label.set_fontsize(plt.rcParams["axes.labelsize"]*scale)
    if ax_inset.get_title():
        ax_inset.title.set_fontsize(base_font_size*scale)
    legend = ax_inset.get_legend()
    if legend is not None:
        plt.setp(legend.get_texts(), fontsize=base_font_size*scale)

def set_colorbar_fonts(cbar, scale=1.0):
    labelsize = plt.rcParams["axes.labelsize"]
    ticksize  = plt.rcParams["xtick.labelsize"]
    titlesize = plt.rcParams["axes.titlesize"]
    # Tick labels
    cbar.ax.tick_params(labelsize=ticksize*scale)
    # Axis labels
    cbar.ax.xaxis.label.set_fontsize(labelsize*scale)
    cbar.ax.yaxis.label.set_fontsize(labelsize*scale)
    # Title (if used via cbar.ax.set_title(...))
    cbar.ax.title.set_fontsize(titlesize*scale)


# # Used for labelling panels in plots (written by AI):

def _corner_specs(corner):
    """
    Return (x, y, ha, va) for a given corner/side/center location.
    Coordinates are in Axes fraction units [0,1].
    """
    corner_map = {
        # Top row
        'upper left':   (0.0, 1.0, 'left',   'top'),
        'upper center': (0.5, 1.0, 'center', 'top'),
        'upper right':  (1.0, 1.0, 'right',  'top'),

        # Middle row
        'center left':  (0.0, 0.5, 'left',   'center'),
        'center':       (0.5, 0.5, 'center', 'center'),
        'center right': (1.0, 0.5, 'right',  'center'),

        # Bottom row
        'lower left':   (0.0, 0.0, 'left',   'bottom'),
        'lower center': (0.5, 0.0, 'center', 'bottom'),
        'lower right':  (1.0, 0.0, 'right',  'bottom'),
    }
    if corner not in corner_map:
        raise ValueError(f"corner must be one of {list(corner_map.keys())}")
    return corner_map[corner]

def _auto_offset_px(ha, va, font_size, base=None):
    """
    Compute a sensible pixel offset given text alignment.
    - ha: 'left'|'center'|'right'
    - va: 'top'|'center'|'bottom'
    The magnitude defaults to ~0.6*font_size unless `base` is provided.
    Signs are chosen so text nudges inward from the axes edge.
    """
    # magnitude in pixels
    mag = (0.2 * font_size) if base is None else float(base)

    # horizontal: left -> +, right -> -, center -> 0
    if ha == 'left':
        dx = +mag
    elif ha == 'right':
        dx = -mag
    else:
        dx = 0.0

    # vertical: top -> -, bottom -> +, center -> 0
    if va == 'top':
        dy = -mag
    elif va == 'bottom':
        dy = +mag
    else:
        dy = 0.0

    return dx, dy

def add_panel_label(ax, label, corner='upper left', offset='auto',
                    font_size=None, **kwargs):
    """
    Place a label (e.g., '(a)') inside a subplot at one of nine positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target subplot.
    label : str
        Label text, e.g. '(a)'.
    corner : str
        One of: 'upper left','upper center','upper right',
                'center left','center','center right',
                'lower left','lower center','lower right'.
    offset : 'auto' | (dx_px, dy_px) | float
        - 'auto' (default): choose a sensible pixel offset based on `corner`
          and `font_size`.
        - (dx, dy): explicit pixel offset.
        - float: use this as the base magnitude for auto-offset (signs still
          depend on corner). E.g., offset=4 -> like 'auto' but ~4 px.
    font_size : float | None
        Font size; defaults to rcParams["axes.labelsize"].
    **kwargs :
        Forwarded to `ax.text` (e.g., fontweight, color, zorder, bbox).
    """
    if font_size is None:
        font_size = plt.rcParams["axes.labelsize"]

    x, y, ha, va = _corner_specs(corner)

    style = dict(fontsize=font_size, ha=ha, va=va)
    style.update(kwargs)

    # Determine offset in pixels
    if offset == 'auto':
        dx, dy = _auto_offset_px(ha, va, font_size)
    elif isinstance(offset, (int, float)):
        # Scalar magnitude but with auto signs
        dx, dy = _auto_offset_px(ha, va, font_size, base=float(offset))
    else:
        # Assume explicit (dx, dy)
        dx, dy = offset

    # Convert pixel offset to display using figure DPI
    trans = ax.transAxes + mtransforms.ScaledTranslation(
        dx / 72.0, dy / 72.0, ax.figure.dpi_scale_trans
    )

    ax.text(x, y, label, transform=trans, **style)

def _prime_string(k):
    return "'" * k if k > 0 else ''

def _letters_with_primes(n, start_index=0):
    letters = list(string.ascii_lowercase)
    labels = []
    for i in range(start_index, start_index + n):
        block = i // 26
        idx = i % 26
        labels.append(letters[idx] + _prime_string(block))
    return labels

def _format_labels(n, start_index=0, wrap='({})'):
    base = _letters_with_primes(n, start_index=start_index)
    return [wrap.format(s) for s in base]

def _iter_axes_in_order(axs, order='row-major'):
    if isinstance(axs, np.ndarray):
        if axs.ndim == 1:
            for ax in axs: yield ax
        elif axs.ndim == 2:
            r, c = axs.shape
            if order == 'row-major':
                for i in range(r):
                    for j in range(c):
                        yield axs[i, j]
            elif order == 'col-major':
                for j in range(c):
                    for i in range(r):
                        yield axs[i, j]
            else:
                raise ValueError("order must be 'row-major' or 'col-major'")
        else:
            for ax in axs.flat: yield ax
    else:
        for ax in axs: yield ax

def _parse_start_label(start_label):
    """
    Convert e.g. 'a' -> index 0, 'c' -> 2, 'a'' -> 26, 'b'' -> 27, etc.
    """
    if not start_label:
        return 0
    # strip parentheses if present
    s = start_label.strip('()')
    # count primes
    base = s.rstrip("'")
    primes = len(s) - len(base)
    idx = string.ascii_lowercase.index(base)
    return idx + 26 * primes

def label_subplots(
    axs,
    corner='upper left',
    offset='auto',
    order='row-major',
    start_index=None,
    start_label=None,
    labels=None,
    wrap='({})',
    text_kwargs=None,
):
    """
    Auto-label a grid/list of subplots.

    Parameters
    ----------
    axs : np.ndarray[Axes] or list[Axes]
    corner : str
        Corner for labels.
    offset : (dx, dy)
        Pixel offset.
    order : {'row-major','col-major'}
    start_index : int or None
        Numeric index to start labeling (0 = a, 25 = z, 26 = a').
    start_label : str or None
        Start at this label directly, e.g. '(c)' or 'b''.
        Overrides start_index if both provided.
    labels : list[str] or None
        Provide your own labels explicitly.
    wrap : str
        Format for wrapping labels (default '({})').
    text_kwargs : dict
        Extra args to ax.text.

    Returns
    -------
    mapping : dict
        {ax: label_text}
    """
    axes_list = list(_iter_axes_in_order(axs, order=order))
    n = len(axes_list)

    if labels is None:
        if start_label is not None:
            idx0 = _parse_start_label(start_label)
        else:
            idx0 = start_index or 0
        lab_seq = _format_labels(n, start_index=idx0, wrap=wrap)
    else:
        if len(labels) != n:
            raise ValueError("len(labels) must match number of axes to label.")
        lab_seq = labels

    mapping = {}
    for ax, lab in zip(axes_list, lab_seq):
        add_panel_label(ax, lab, corner=corner, offset=offset, **(text_kwargs or {}))
        mapping[ax] = lab
    return mapping

def get_panel_label(ax, axs, order='row-major',
                    start_index=0, start_label=None,
                    wrap='({})'):
    """
    Return the appropriate label string for a given subplot axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The specific subplot axis to label.
    axs : array/list of Axes
        The array/list returned by plt.subplots.
    order : {'row-major','col-major'}
        Ordering of labels.
    start_index : int
        Start index (0 = a, 25 = z, 26 = a').
    start_label : str or None
        Explicit starting label (overrides start_index).
    wrap : str
        Format string for wrapping label (default '({})').

    Returns
    -------
    label : str
        Label string for the given axis (e.g. '(a)', '(c)', '(b')').
    """
    axes_list = list(_iter_axes_in_order(axs, order=order))
    n = len(axes_list)

    # Which position is this axis?
    try:
        idx = axes_list.index(ax)
    except ValueError:
        raise ValueError("The given axis is not in the provided axes array.")

    # Starting index
    if start_label is not None:
        # Reuse the parser from earlier code
        s = start_label.strip('()')
        base = s.rstrip("'")
        primes = len(s) - len(base)
        idx0 = string.ascii_lowercase.index(base) + 26 * primes
    else:
        idx0 = start_index

    # Build labels
    labels = _format_labels(n + idx0, start_index=idx0, wrap=wrap)
    return labels[idx]


#Some helper functions for the tutorial:
#
# Generate a list of constrained or typical surrogates.
# Constrained surrogates are generated successively, as in the analysis notebooks;
# typical surrogates are generated independently from the original input sequence.
def gen_surrogate_list(val_seq, model='powerlaw', method='constrained', num_surr=1, num_trans=None, lower_cutoff_hat=None, param_hat=None):
    if (lower_cutoff_hat == None):
        lower_cutoff_hat = np.floor(min(val_seq))

    surr_val_seq_list = []

    if (method == 'constrained'):
        surr_val_seq = list(val_seq)
        for i_surr in range(num_surr):
            surr_val_seq = gen_surrogate(surr_val_seq, model=model, num_trans=num_trans, method=method, lower_cutoff_hat=lower_cutoff_hat, param_hat=param_hat)
            random.shuffle(surr_val_seq)
            surr_val_seq_list = surr_val_seq_list + [list(surr_val_seq)]

    elif (method == 'typical'):
        if (param_hat == None):
            param_hat, _ = fit_model(val_seq, model=model, lower_cutoff_hat=lower_cutoff_hat)
        for i_surr in range(num_surr):
            surr_val_seq = gen_surrogate(val_seq, model=model, num_trans=num_trans, method=method, lower_cutoff_hat=lower_cutoff_hat, param_hat=param_hat)
            surr_val_seq_list = surr_val_seq_list + [list(surr_val_seq)]

    else:
        raise Exception('method=' + str(method) + ' not recognised.')

    return surr_val_seq_list


# Calculate the rank-based quantile used for hypothesis testing in the analysis notebooks.
def calc_surrogate_quantile(obs_stat, surr_stat_list):
    stat_val_surr_list = np.array(surr_stat_list)
    num_surr = len(stat_val_surr_list)

    abs_obs_stat = abs(obs_stat)#Add small random perturbations to avoid ties
    if (abs_obs_stat == np.inf):
        abs_obs_stat = 0
    obs_stat = obs_stat + 10**-6*abs_obs_stat*(np.random.uniform() - 0.5)
    stat_val_surr_list = stat_val_surr_list + 10**-6*abs_obs_stat*(np.random.uniform(size=num_surr) - 0.5)

    rankMin = 1 + sum(stat_val_surr_list < obs_stat)
    rankMax = 1 + sum(stat_val_surr_list <= obs_stat)
    r = random.randint(rankMin, rankMax)
    q = (r - 0.5)/(num_surr + 1)

    return q

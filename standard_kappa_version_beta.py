#
# Name:
#   standard_kappa
# Purpose:
#   Get Maxwellian decomposition coefficients for the standard kappa distribution
# Author:
#   Chengcai Shen
# Update:
#   2025-04-08: version beta
#

#
# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------
# Standard library imports
import os
import sys
import math
import pickle
from datetime import datetime
import multiprocessing

# Third-party library imports
import numpy as np
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
from joblib import parallel_backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class BreakLoop(Exception):
    pass


def func_te_range(kappa_in):
    """
    Determines the range and number of points for a parameter based on the input kappa value.

    This function calculates the start (`tes`), end (`tee`), and the number of points (`n`) 
    for a range of values based on the input `kappa_in`. The range and number of points 
    are determined by specific conditions applied to `kappa_in`.

    Parameters:
        kappa_in (float): The input kappa value used to determine the range and number of points.

    Returns:
        list: A list containing three elements:
            - tes (float): The starting value of the range.
            - tee (float): The ending value of the range.
            - n (int): The number of points in the range, calculated as 
              `n = int((tee - tes) / 0.0008) + 1`.
    """
    # set fitting parameters
    tes = -1.0
    tee = 3.0
    if kappa_in <= 3.0:
        tes = -2.0
        tee = 4.5
    if kappa_in >= 100.0:
        tes = -0.5
        tee = 0.5

    n = int((tee - tes) / 0.0008) + 1
    return [tes, tee, n]


def func_fit_single(x, y, sw_positive, n_tempoint, tempoint_s, tempoint_e):
    """
    Fits a model to the given data and returns the fitting coefficients.

    This function performs a fitting operation on the provided data using
    the `func_fit` method. It calculates the number of temperature
    points based on the given range and step size, and handles any exceptions
    that may occur during the fitting process.

    Parameters:
        x (array-like): The independent variable data for fitting.
        y (array-like): The dependent variable data for fitting.
        sw_positive (integer): A flag indicating whether to enforce positivity
            constraints during the fitting process: sw_positive = 1 for positive,
                                                                = 0 for non-positive.
        n_tempoint (integer): The number of temperature sampling-points.
        tempoint_s (float): The starting point of the temperature range.
        tempoint_e (float): The ending point of the temperature range.

    Returns:
        tuple:
            c_i (float): The coefficient `c_i` resulting from the fitting process.
                         Returns -1.0e5 if an error occurs.
            a_i (float): The coefficient `a_i` resulting from the fitting process.
                         Defaults to 1.0 if an error occurs.

    Raises:
        Exception: Any exception raised during the fitting process is caught
                   and logged, but the function will return default values
                   for `c_i` and `a_i` in such cases.
    """

    try:
        y_predict, c_i, a_i = func_fit(x, y,
                                    sw_positive=sw_positive,
                                    tempoint_s=tempoint_s,
                                    tempoint_e=tempoint_e,
                                    n_tempoint=n_tempoint)
    except Exception as e:
        print(f"An error occurred: {e}")
        c_i = 0.0
        a_i = 1.0

    return c_i, a_i


def func_fit(x, y, sw_positive=1, n_tempoint=101, tempoint_s=-2, tempoint_e=5, tempoint_array=None):
    """
    Perform a linear regression fit using a set of temperature-dependent basis functions.
    Parameters:
    -----------
    x : array-like
        Input data for the independent variable.
    y : array-like
        Target data for the dependent variable.
    sw_positive : int, optional, default=1
        If set to 1, enforces non-negative coefficients in the linear regression.
        If set to 0, allows coefficients to take any value.
    n_tempoint : int, optional, default=101
        Number of temperature sampling points to generate if `tempoint_array` is not provided.
    tempoint_s : float, optional, default=-2
        Start exponent for the log-scale temperature sampling points.
    tempoint_e : float, optional, default=5
        End exponent for the log-scale temperature sampling points.
    tempoint_array : array-like, optional, default=None
        User-defined array of temperature sampling points. If provided, it overrides
        the log-scale generation of temperature points.
    Returns:
    --------
    y_pre : array-like
        Predicted values of the dependent variable based on the linear regression model.
    c_i : list
        Coefficients of the linear regression model corresponding to each temperature basis function.
    a_i : list
        Temperature sampling points used as basis functions in the regression.
    Notes:
    ------
    - The function generates temperature-dependent basis functions using the `func_maxwell_e` function.
    - Linear regression is performed using scikit-learn's `LinearRegression` class.
    - Parallel threading is employed for efficiency during the regression process.
    """
    
    # Set the temperature sampling points
    if tempoint_array is not None:
        # User define t_coff_list
        t_coeff_list = np.array(tempoint_array)
    else:
        # case 1: log-scale
        t_coeff_list = np.logspace(tempoint_s, tempoint_e, n_tempoint)

     
    for ite in range(len(t_coeff_list)):
        kbt_c = t_coeff_list[ite]
        
        X2 = func_maxwell_e(x, kbt_c).reshape(-1,1)
        if (ite == 0):
            X = X2
        else:
            X = np.hstack((X, X2))
    
    # Employ linear regression
    with parallel_backend('threading'):
        if (sw_positive == 0):
            lr2 = LinearRegression(n_jobs=-1)
        else:
            lr2 = LinearRegression(positive=True, n_jobs=-1)
        lr2.fit(X, y)
    y_pre = lr2.predict(X)
    
    # Get coefficients
    c_i = []
    a_i = []
    for ite in range(len(t_coeff_list)):
        c_i.append(lr2.coef_[ite])
        a_i.append(t_coeff_list[ite])
        
    return y_pre, c_i, a_i

def func_maxwell_e(e, kbt):
    p0 = 2.0/math.sqrt(math.pi)
    p1 = (1./kbt)**1.5
    p2 = np.sqrt(e)
    p3 = np.exp(-e/kbt)
    f = p0*p1*p2*p3
    f_min = 0.0
    f[f <= f_min] = f_min
    return f

def func_sum_mutiple_maxwell(e, c, a):
    ncom = len(c)
    fk = np.zeros(len(e))
    for icom in range(ncom):
        fi = c[icom]*func_maxwell_e(e, a[icom])
        fk += fi
    return fk

def func_ak(kappa):
    #return gamma(kappa+1.0)/(gamma(kappa-0.5)*(kappa-1.5)**1.5)
    a = np.log(gamma(kappa+1.0))
    b = np.log(gamma(kappa-0.5))
    c = np.log((kappa-1.5)**1.5)
    s = a - b - c
    return np.exp(s)


def func_kappa_e(e, kbt, kappa):
    Ak = func_ak(kappa)
    p0 = 2.0/math.sqrt(math.pi)
    p1 = (1./kbt)**1.5
    p2 = np.sqrt(e)
    p3 = (1.0 + e/((kappa-1.5)*kbt))**(-(kappa+1.0))
    return Ak*p0*p1*p2*p3

# -----------------------------------------------------------------------------
# Main loop: Kappa-distribution
# -----------------------------------------------------------------------------
# Check if the folder './data_temp' exists, and create it if it doesn't
folder_path = './data_temp'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


kappa_list = [1.7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 100]
print(kappa_list)

nsample = 61
kbt = 1.0
ekbt = np.logspace(-2.0, 4.0, nsample)

# output file
now = datetime.now()
dtime_str = now.strftime("%Y%m%d_%H%M%S")

# enter the kappa loops
for ik in range(len(kappa_list)):
    kappa_in = kappa_list[ik]
    print(f"kappa = {kappa_in}")
    
    # set fitting parameters:
    tes, tee, nte = func_te_range(kappa_in)

    # set the kappa distribution
    f_e = func_kappa_e(ekbt, kbt, kappa_in)

    # perform the fitting
    # f_e_pre, c_i, a_i = func_fit(ekbt, f_e, n_tempoint=n, tempoint_s=tes, tempoint_e=tee)
    c_i, a_i = func_fit_single(ekbt, f_e, 1.0, nte, tes, tee)

    # get the sum of multiple maxwellian
    f_sum_maxwell = func_sum_mutiple_maxwell(ekbt, c_i, a_i)
    

    #
    # save into array
    #
    if (ik == 0):
        ci_out = [[]]
        ai_out = [[]]
    ci_out.append(c_i)
    ai_out.append(a_i)
    # print(a_i[0], a_i[-1])

    #
    # plot to figure
    #
    c_i = np.array(c_i)
    a_i = np.array(a_i)

    e_plot = np.logspace(-2.0, 4.0, nsample)
    f_kappa = func_kappa_e(e_plot, kbt, kappa_in)
    f_fit = func_sum_mutiple_maxwell(e_plot, c_i, a_i)

    fig = plt.figure(figsize=(12, 4))
    text_sum = '$f_{fit}$' + ' = ' + '${\\sum}$' + '$c_{i} \\times f_{Maxwell}(a_{i} \\times e)}$'
    text_kappa = '$f_\kappa$ = {0:.1f}'.format(kappa_in)
    text_error = '${(f_{fit} - f_{\kappa})/f_{\kappa}}$'

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(e_plot, f_kappa, color='black', label=text_kappa)
    ax.plot(e_plot, f_fit, ls='--', color='orange', label=text_sum)

    ax.set_title('(a) Kappa Distribution Profile')
    ax.set_xlabel('e (k$_B$T)')
    ax.set_ylabel('f(e) $\\times\ k_BT$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1.0e-16, 10.0])
    ax.legend()

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(e_plot, np.fabs(f_fit - f_kappa) / f_kappa, '.', color='orange')
    ax.set_title('(b) Relative Error')
    ax.set_xlabel('e (k$_B$T)')
    ax.set_ylabel(text_error)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(a_i, c_i, '.', c='orange')
    ax.set_title('(c) Maxwellian Components')
    ax.set_xlabel('$a_i$')
    ax.set_ylabel('$c_i$')
    ax.set_xscale('log')
    #ax.set_ylim([0, np.amax(c_i)])

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('./data_temp/fig_all_kappa_{0:.1f}_{1:s}.png'.format(kappa_in, dtime_str), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------------
    # save results
    # -----------------------------------------------------------------------------
    outfile = './data_temp/sav_kappa_{0:.1f}_{1:s}.p'.format(kappa_in, dtime_str)
    pickle.dump([e_plot, kappa_in, c_i, a_i], open(outfile, 'wb'))

# -----------------------------------------------------------------------------
# Save all in one
# -----------------------------------------------------------------------------
outfile = './data_temp/coeff_ci_ai_{0:s}.p'.format(dtime_str)
pickle.dump([e_plot, kappa_list, ci_out, ai_out], open(outfile, 'wb'))
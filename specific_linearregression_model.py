'''
The Main Application Process
=====================================
This script demonstrates the application of the `MaxwellianDeepFit` class to fit a kappa electron energy distribution 
using a neural network-based approach. The process involves the following steps:
1. **Set the Target Kappa Distribution**:
    - Define the target kappa distribution using the `func_kappa_e` function.
    - Generate input energy values (`x_fit_input`) and corresponding target distribution values (`y_fit_input`).
   **Prepare Output Folder**:
    - Check if the folder `./epoch_animation` exists. If not, create it to store diagnostic plots during training.
   **Initialize Neural Network**:
    - Define hyperparameters such as learning rate (`lrate`), number of Maxwellian components (`nparams`), and training epochs (`epochs`).
    - Generate initial guesses for Maxwellian component parameters (`kbt_guess` and `ws_guess`).
    - Normalize and sort the initial guesses for better training stability.
2. **Train the Neural Network**:
    - Instantiate the `MaxwellianDeepFit` class with the specified hyperparameters and initial guesses.
    - Append an additional value to the target distribution (`y_fit_input_ext`) for training.
    - Train the model using the `train` method, which optimizes the Maxwellian components to fit the target distribution.
   **Extract and Save Results**:
    - Extract the trained Maxwellian component weights (`ci_arr`) and parameters (`ai_arr`).
    - Compute the predicted distribution (`y_fit_predict`) using the trained model.
    - Save the results, including hyperparameters, input data, trained parameters, and training metrics, to a pickle file (`output_test.p`).
Key Functions and Classes:
--------------------------
- `func_kappa_e`: Computes the kappa electron energy distribution for given energy, temperature, and kappa parameter.
- `MaxwellianDeepFit`: Implements the neural network-based fitting model for Maxwellian distributions.
- `train`: Trains the neural network to fit the target distribution.
- `func_plot_training`: Generates diagnostic plots during training to visualize the fitting progress.
Dependencies:
- scipy
- pickle
- os etc.

Author:
--------
- Chengcai Shen

Update History:
----------------
- Initial version created on 2025-04-08.
- Updated on 2025-04-15: Added detailed comments and improved documentation.
'''

# Standard library imports
import os
import sys  
import pickle  
from datetime import datetime  
import math 

# Third-party library imports
import numpy as np  
from scipy.special import gamma  # Provides special mathematical functions, like the gamma function

# PyTorch imports
import torch  
import torch.nn as nn  # Neural network module in PyTorch
import torch.optim as optim  # Optimization algorithms in PyTorch

# Matplotlib imports
import matplotlib.pyplot as plt  

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 120

torch.set_default_dtype(torch.float32)
if torch.backends.mps.is_available():
    torch.device("mps")
else:
    print("MPS device not found.")

fmin_floor = 1.0e-20

def func_gaussian(x, mu, sig):
    return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def normalize_log(y):
    y[y <= fmin_floor] = fmin_floor
    log_min = np.log10(fmin_floor)
    y_log = (np.log10(y) - log_min)/(-log_min)
    return y_log

def normalize_kbt(kbt_in):
    kbt_out = (np.log10(kbt_in) - (-5.0)) / 10.0
    return kbt_out

def denormalize_kbt(kbt_in):
    kbt_out = 10.0 ** (kbt_in * 10.0 + (-5.0))
    return kbt_out

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

class MaxwellianDeepFit:
    """
    MaxwellianDeepFit
    =================
    This module implements a deep learning-based fitting model for Maxwellian electron energy distributions. 
    The `MaxwellianDeepFit` class is designed to fit a given electron energy distribution using a combination 
    of Maxwellian components.
    Classes
    -------
    MaxwellianDeepFit
        A class that encapsulates the Maxwellian fitting model, training process, and evaluation utilities.
        Methods
        -------
        __init__(epochs=500000, learning_rate=0.0001, num_params=200, kbt_guess=None, ws_guess=None)
            Initializes the MaxwellianDeepFit model with specified hyperparameters and initial guesses.
        train(x, y)
            Trains the model on the given input data `x` and target data `y`.
        MeanRatioLoss(output, target)
            Computes a weighted mean ratio loss between the predicted and target values.
        func_maxwell_e(e, kbt)
            Computes the Maxwellian electron energy distribution for a given energy `e` and temperature `kbt`.
        func_plot_training(epoch, x, y, y_pred)
            Generates and saves diagnostic plots during training, including:
            - Input vs. predicted distributions
            - Relative differences
            - Maxwellian component weights and parameters.
    Inner Classes
    -------------
    MaxwellianModel
        A PyTorch neural network module that represents the Maxwellian fitting model.
        Methods
        -------
        __init__(num_params, kbt_guess=None, ws_guess=None)
            Initializes the model with the specified number of Maxwellian components and optional initial guesses.
        forward(x)
            Performs a forward pass through the model, computing the predicted distribution for input `x`.
    Attributes
    ----------
    epochs : int
        The number of training epochs. Default is 500,000.
    learning_rate : float
        The learning rate for the optimizer. Default is 0.0001.
    num_params : int
        The number of Maxwellian components to use in the fitting model. Default is 200.
    model : MaxwellianModel
        The PyTorch model representing the Maxwellian fitting process.
    optimizer : torch.optim.Adam
        The optimizer used for training the model.
    loss_fn : callable
        The loss function used during training. Default is `MeanRatioLoss`.
    epoch_losses : list
        A list to store the loss values at each epoch during training.
    epoch_times : list
        A list to store the timestamps of each epoch during training.
    Usage
    -----
    1. Initialize the `MaxwellianDeepFit` class with desired hyperparameters:
        >>> model = MaxwellianDeepFit(epochs=10000, learning_rate=0.001, num_params=100)
    2. Train the model with input data `x` and target data `y`:
        >>> model.train(x, y)
    3. Use the diagnostic plots generated during training to evaluate the fitting process.
    Notes
    -----
    - The model uses a custom loss function (`MeanRatioLoss`) that incorporates logarithmic scaling and 
      weighting based on the target distribution.
    - The training process stops early if the relative error between predictions and targets falls below 
      a specified threshold (`error_rellimt`).
    - Diagnostic plots are saved during training to visualize the fitting progress and Maxwellian components.
    Dependencies
    ------------
    - numpy
    - matplotlib
    - torch
    - datetime
    """
    class MaxwellianModel(nn.Module):

        def __init__(self, num_params, kbt_guess=None, ws_guess=None):
            super().__init__()

            if kbt_guess is None:
                kbt_guess = np.sort(np.random.uniform(0, 1.0, num_params))
            if ws_guess is None:
                ws_guess = np.random.uniform(0, 1.0, num_params)

            kbt_tensor = torch.tensor(kbt_guess, dtype=torch.float32)
            ws_tensor = torch.tensor(ws_guess, dtype=torch.float32)

            self.kbt = nn.Parameter(kbt_tensor)
            self.ws = nn.Parameter(ws_tensor)
            self.num_params = num_params

        def forward(self, x):
            ws_floor = -1.0e-7
            self.ws.data = torch.clamp(self.ws.data, ws_floor, 1.0)
            
            kbt_norm = self.kbt
            ws = self.ws
            y = torch.zeros_like(x)
            for w, kbt_norm_val in zip(ws, kbt_norm):
                kbt = denormalize_kbt(kbt_norm_val)
                f = 2.0 / math.sqrt(math.pi) * (1. / kbt) ** 1.5 * x ** 0.5 * torch.exp(-x / kbt)
                y += w * f

            sumci = np.sum(ws.detach().numpy())
            sumci_tensor = torch.tensor([[sumci]], dtype=torch.float32)
            y_predict = torch.cat((y, sumci_tensor), dim=0)

            return y_predict

    def __init__(self, epochs=500000, learning_rate=0.0001, num_params=200,
                 kbt_guess=None, ws_guess=None):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_params = num_params

        self.model = self.MaxwellianModel(self.num_params, kbt_guess=kbt_guess, ws_guess=ws_guess)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_fn = self.MeanRatioLoss

        self.epoch_losses = []
        self.epoch_times = []

    def MeanRatioLoss(self, output, target):
        output_copy = torch.clone(output)
    
        ratio = output_copy/target
        res = torch.where(output_copy < 0.0)
        ratio[res] = 10.0*torch.abs(ratio[res])
 
        ratio_log = torch.abs(torch.log10(ratio))

        target_log = torch.log10(torch.clone(target))
        t_log_max = torch.max(target_log)
        t_log_min = torch.min(target_log) - 1.0
        target_norm = (target_log - t_log_min)/(t_log_max - t_log_min)

        weights = target_norm
        loss = torch.mean(ratio_log*weights)
        return loss

    def train(self, x, y):

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_tensor)

            error_rellimt = 1.0e-2
            error_rel = torch.max(torch.abs(y_pred - y_tensor)/y_tensor)
            if error_rel < error_rellimt:
                print(f"Stopping training at epoch {epoch + 1} because the relative error approaches the threshold.")
                res = self.func_plot_training(epoch, x_tensor, y_tensor, y_pred)
                break

            loss = self.loss_fn(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 1000 == 0:
                learning_rate = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}/{self.epochs}, Loss: {loss.item()}, lr: {learning_rate}, Error_rel: {error_rel:.3e}')

                self.epoch_losses.append(loss.item())
                now = datetime.now()
                current_time_second = (now - datetime(1970, 1, 1)).total_seconds()
                self.epoch_times.append(current_time_second)

                res = self.func_plot_training(epoch, x_tensor, y_tensor, y_pred)

    def func_maxwell_e(self, e, kbt):
        p0 = 2.0 / math.sqrt(math.pi)
        p1 = (1. / kbt) ** 1.5
        p2 = e ** 0.5
        p3 = np.exp(-e / kbt)
        f = p0 * p1 * p2 * p3
        return f

    def func_plot_training(self, epoch, x, y, y_pred):
        y_input_ext = y.detach().numpy()
        y_input = y_input_ext[:-1]
        y_predict_ext = y_pred.detach().numpy()
        y_predict = y_predict_ext[:-1]
        x_input = x.detach().numpy()

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(x_input, y_input, color='C0', label='Input')
        ax1.plot(x_input, y_input, '.', color='C0')
        ax1.plot(x_input, y_predict, '.', color='C1', label='Fitting')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim([0.5*np.amin(y_input), 1.0])
        ax1.set_xlabel('E(k$_B$T)')
        ax1.set_ylabel('f(E) $\\times$ k$_B$T')
        ax1.set_title('(a) Electron distirubtions at Epoch={0:d}'.format(epoch))
        ax1.legend()
        ax1.grid('True')

        ax2 = fig.add_subplot(1, 3, 2)
        diff = np.fabs(y_predict - y_input)
        rediff = diff/y_input
        ratio = y_predict/y_input
        res = np.where(y_predict < 0.0)
        ratio[res] = 10.0 + np.fabs(ratio[res])
        ratio_log = np.fabs(np.log10(ratio))
        ax2.plot(x_input, rediff, 'o', color='C1')
        ax2.set_ylim([1.0e-7, 1.0e2])
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('E(k$_B$T)')
        ax2.set_ylabel('$|f_{Fitting}-f_{Input}|/f_{Input}$')
        ax2.set_title(f'(b) Relative Difference')
        ax2.grid('True')

        ax3 = fig.add_subplot(1, 3, 3)
        ci_arr = [ws.item() for ws in network.model.ws.detach().numpy()]
        ai_arr_norm = [kbt.item() for kbt in network.model.kbt.detach().numpy()]
        ai_arr = denormalize_kbt(np.array(ai_arr_norm))

        ax3.plot(ai_arr, ci_arr, '.', color='C1')
        ax3.set_xlim([1.0e-3, 1.0e3])
        ax3.set_xscale('log')
        ax3.set_xlabel('$a_i$')
        ax3.set_ylabel('$c_i$')
        ax3.set_title(f'(c) Maxwellian Components')
        ax3.text(0.05, 0.90, f'$\\sum(c_i)$ = {np.sum(ci_arr):.3f}', transform=ax3.transAxes)
        ax3.grid('True')

        plt.subplots_adjust(left=0.075, right=0.99, wspace=0.3)
        plt.savefig(f'./epoch_animation/fig_epoch_{epoch:07d}.png', dpi=150)
        plt.close()
        return 1


# ---------------------------------------------------------------------------------------------------------------------
# 1: Set the target kappa distribution
# ---------------------------------------------------------------------------------------------------------------------

kappa_in = 6.0
x_fit_input = np.logspace(-2, 2, 24)
y_fit_input = np.zeros(len(x_fit_input))
numb = len(x_fit_input)
for i in range(numb):
    e = x_fit_input[i] 
    y_fit_input[i] = func_kappa_e(e, 1.0, kappa_in)

# Check if the folder exists
folder_path = "./epoch_animation"
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

# -----------------------------------------------------------------------------
# 2: Employ the neural network to fit the kappa distribution
# -----------------------------------------------------------------------------
lrate = 1.0e-5
nparams = 32
epochs = 1000000
kbt_guess = np.linspace(0.3, 0.7, nparams)
index_ws = np.linspace(1, nparams, nparams)
ws_guess = 0.0*func_gaussian(index_ws, 0.5*nparams, 0.5*nparams) + np.random.uniform(0, 1.0e-7, nparams)

index_sort = np.argsort(kbt_guess)
kbt_guess = kbt_guess[index_sort]
ws_guess = ws_guess[index_sort]
ws_guess /= np.sum(ws_guess)
nparams = len(ws_guess)
#print(ai_chosen)

torch.manual_seed(0)
network = MaxwellianDeepFit(epochs=epochs,
                            learning_rate=lrate,
                            num_params=nparams,
                            kbt_guess=kbt_guess,
                            ws_guess=ws_guess)

y_fit_input_ext = np.append(y_fit_input, 1.0)
network.train(x_fit_input, y_fit_input_ext)

ci_arr = [ws.item() for ws in network.model.ws.detach().numpy()]
ai_arr_norm = [kbt.item() for kbt in network.model.kbt.detach().numpy()]
ai_arr = denormalize_kbt(np.array(ai_arr_norm))
loss_epchos = network.epoch_losses
time_epchos = network.epoch_times
time_epchos = np.array(time_epchos) - time_epchos[0]

y_fit_predict = np.zeros_like(x_fit_input)
for ci, ai in zip(ci_arr, ai_arr):
    y_fit_predict += ci * network.func_maxwell_e(x_fit_input, ai)

file = open('output_test.p', 'wb')
pickle.dump([lrate, nparams, epochs, x_fit_input, y_fit_input, ci_arr, ai_arr, loss_epchos, time_epchos], file)
file.close()

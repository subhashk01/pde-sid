from sympy import symbols, exp, diff, lambdify
import numpy as np
import matplotlib.pyplot as plt
import math
from configs import Config
from util import read_bases

config = Config()


def gaussians(plot = True, pos = False):
    # generates u using a gaussian mixture model. returns array of form [1+num_derivs, N_p]

    # N_g = number of gaussians in our mixture model
    # N_p the number of points to evaluate each gaussian at between -10,10
    # num_derivs is the number of derivatives to take

    x_vals = np.linspace(config.x_min, config.x_max, config.N_p)
    y_vals = np.zeros((config.num_derivs+1, config.N_p))

    for _ in range(config.N_g):
        mu_val = np.random.uniform(-3, 3)
        sigma_val = 1.5
        if pos:
            A = np.random.uniform(1, 5)
        else:
            A = np.random.uniform(-5, 5)

        x, mu, sigma= symbols('x mu sigma')
        # Define Gaussian function
        gaussian = exp(-((x - mu)**2) / (2 * sigma**2))

        # Calculate up to 3rd derivative (you can go higher)
        derivatives = [gaussian]
        for i in range(1, 1+config.num_derivs):
            derivatives.append(diff(derivatives[-1], x))

        # Substitute values for mu and sigma

        params = {mu: mu_val, sigma: sigma_val}

        # Evaluate derivatives at N_p x-points
    
        numerical_derivatives = [lambdify(x, d.subs(params), 'numpy') for d in derivatives]
        for i,d in enumerate(numerical_derivatives):
            y_vals[i] += A/(math.sqrt(2*math.pi)*sigma_val) * d(x_vals)

    if plot:
        plt.plot(x_vals, y_vals[0], label  = 'u'+f' sum = {np.sum(y_vals[i]):.1e}')
        for i in range(1, y_vals.shape[0]):
            y_sum = np.sum(y_vals[i])
            plt.plot(x_vals, y_vals[i], label  = 'u_'+'x'*i+f' sum = {y_sum:.1e}')
        plt.legend()
        plt.show()

    if pos:
        assert np.all(y_vals[0]>=0), f"y_vals has negative values when pos is {pos}"
    return x_vals, y_vals



def generate_us(var_num = '', pos = False):
    y_vals = []
    for i in range(config.num_curves):
        _, y_val = gaussians(plot = False, pos = pos)
        y_vals.append(y_val)
        if i%5 == 0:
            print(var_num,i)
    y_vals = np.array(y_vals)
    assert y_vals.shape == (config.num_curves, config.num_derivs+1, config.N_p), "y_vals has wrong shape"
    if pos:
        assert np.all(y_vals[:,0,:] >= 0), f"y_vals has negative values when pos is {pos}"
    return y_vals


def generate_variables(pos = False):

    y_vals = []
    for var_num in range(config.max_num_eqs):
        y_vals_var = generate_us(var_num = var_num, pos = pos)
        y_vals.append(y_vals_var)
    y_vals = np.array(y_vals)
    assert y_vals.shape == (config.max_num_eqs, config.num_curves, config.num_derivs+1, config.N_p), "y_vals has wrong shape"
    if pos:
        save_str = 'test_poscurves.npy'
        assert np.all(y_vals[:,:,0,:] >= 0), f"y_vals has negative values when pos is {pos}"
    else:
        save_str = 'test_curves.npy'
    with open(save_str, 'wb') as f:
        np.save(f, y_vals)

# def generate_ordinary():
#     shape = (config.max_num_eqs, config.num_curves, 1, config.N_p)
#     # generate array with random values between 1 and 3 with shape
#     y_vals = np.random.uniform(1, 3, shape)
#     assert y_vals.shape == (config.max_num_eqs, config.num_curves, 1, config.N_p), "y_vals has wrong shape"
#     # save y_vals to file
#     with open('test_ordinary.npy', 'wb') as f:
#         np.save(f, y_vals)

if __name__ == '__main__':
    generate_variables(pos = True)

import torch
import matplotlib.pyplot as plt
from configs import Config
import numpy as np
import math
from sympy import symbols, diff, Function

config = Config()


def gaussians(x_vals, plot=False):
    # Using PyTorch
    
    y_vals = torch.zeros((config.num_derivs+1, config.N_p))
    for _ in range(config.N_g):
        mu_val = torch.tensor(np.random.uniform(-3, 3))
        sigma_val = 1.5
        A = torch.tensor(np.random.uniform(-5, 5))

        # Define Gaussian function
        gaussian = torch.exp(-((x_vals - mu_val)**2) / (2 * sigma_val**2))

        # Store the values
        y_vals[0] += A * gaussian

        # Differentiate and store other derivatives
        for i in range(1, config.num_derivs+1):
            gaussian = torch.autograd.grad(gaussian.sum(), x_vals, create_graph=True)[0]
            y_vals[i] += A * gaussian

    # Convert to numpy for plotting
    x_vals_np = x_vals.detach().numpy()
    y_vals_np = [y.detach().numpy() for y in y_vals]

    if plot:
        for y in y_vals_np:
            plt.plot(x_vals_np, y)
        plt.show()

    return y_vals

def generate_us():
    x_vals = torch.linspace(config.x_min, config.x_max, config.N_p, requires_grad = True)
    y_vals_list = []
    for i in range(config.num_curves):
        y_val = gaussians(x_vals, plot = False)
        y_vals_list.append(y_val.unsqueeze(0))  # Add batch dimension
        if i % 5 == 0:
            print(i)

    # Stack the tensors along the batch dimension
    y_vals_tensor = torch.cat(y_vals_list, dim=0)
    return x_vals, y_vals_tensor


if __name__ == '__main__':
    x, y = generate_us()

    # Compute f
    f = y[0][0] * y[0][1]*y[0][2]
    print(f.shape)
    print
    f_derivative = torch.autograd.grad(f[2], x[2], create_graph=True)[0]    

    # # Compute the derivative of f w.r.t. x
    # for q in range(30):
    #     print(q)
    #     f_derivative = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

    # # Plot f
    # plt.plot(x.detach().numpy(), f.detach().numpy(), label='f')
    
    # # Plot the derivative of f
    # plt.plot(x.detach().numpy(), f_derivative.detach().numpy(), label='df/dx')
    # plt.axhline(y = 0)
    
    # plt.legend()
    # plt.show()

    # print(y.shape)
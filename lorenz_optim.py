from util import create_polynomial_basis, check_function_integral, inverse_spherical_transform
import numpy as np
from model import check_trivial
import matplotlib.pyplot as plt
from optimize import optimize, plot_loss_parameters
import torch, os
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def create_lorenz_bases(max_exp = 3):
    variables = ['x', 'y', 'z']
    bases = create_polynomial_basis(variables, max_exp)
    us = np.load('test_curves.npy')
    with open('bases/lorenz_basis.txt', 'w') as f:
        for base in bases:
            val = check_function_integral(base, us, ['x', 'y', 'z'])
            if not abs(val.mean())<1e-3:
                f.write(base + '\n')
            else:
                print(base, val.mean())

def optimize_lorenz(do = 'run'):
    if do == 'run':
        #fs = ['x_t = {0}*y-{0}*x', 'y_t = {1}*x-{2}*x*z-{3}*y', 'z_t = {4}*x*y-{5}*z']
        fs = ['x_t = {0}*y-{0}*x', 'y_t = {1}*x-{2}*x*z-{3}*y', 'z_t = {4}*x*y-{5}*z']
        optimize(fs, 'lorenz', bases = 'lorenz', sphere = False,epochs = 20000)
    else:
        fs = ['x_t = {0}*y-{0}*x', 'y_t = {1}*x-x*z-y', 'z_t = x*y-{2}*z']
        plot_loss_parameters(fs, 'lorenz', 'lr0.001_A0_B1_epochs20000_cos_nonorm', sphere = False)

def run_many_lorenz(title = 'lorenz',num_runs = 100, num_epochs = 5000):
    # i think this is probably broken rn. normalizing per equation doesn't work for lorenz since there is only one variable for x_t
    lorenz = ['x_t = {0}*y-{0}*x', 
              'y_t = {1}*x-x*z-y', 
              'z_t = x*y-{2}*z']
    print(lorenz)
    vals = []
    for i in range(num_runs):
        fname = f'run{i}_'
        print(fname)
        starting_vals, best_param = optimize(lorenz, lr = .01, fname = fname, title=title, bases='lorenz', epochs=num_epochs, save=True, sphere = False)
        combined = torch.cat((starting_vals.unsqueeze(0), best_param.unsqueeze(0)), dim=0)
        vals.append(combined)
        vals_tensor = torch.stack(vals)
        torch.save(vals_tensor, f'optimize/{title}_3params/{title}_{num_epochs}epochs_{num_runs}runs.pt')
        print(vals_tensor.shape)
    # read in tensor
    vals_tensor = torch.load(f'optimize/{title}_3params/{title}_{num_epochs}epochs_{num_runs}runs.pt')

def look_at_many_lorenz(filename = 'lr0.01_A0_B1_epochs5000_cos_nonorm'):
    # check if file is in directory
    # fs = ['x_t = {0}*y-{0}*x', 
    #       'y_t = {1}*x-{2}*x*z-{2}*y', 
    #       'z_t = {3}*x*y-{4}*z'] #{0} is sigma, {1} is rho, {4} is beta
    fs = ['x_t = {0}*y-{0}*x', 
              'y_t = {1}*x-x*z-y', 
              'z_t = x*y-{2}*z']
    for i in range(100):
        dir = 'optimize'
        fname = f'run{i}_{filename}'
        title = 'lorenz'
        full_file = dir+'/'+title+'_3params'+'/'+fname+'_loss.npy'
        if os.path.exists(full_file):
            plot_loss_parameters(fs, title, fname, sphere=False)

def update_xyz(x,y,z, sigma, rho, beta, step = .01):
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-beta*z
    x_t = x+dxdt*step
    y_t = y+dydt*step
    z_t = z+dzdt*step
    return x_t, y_t, z_t

def plot_lorenz_3d(sigma, rho, beta, starting_vals = [10,0,10], title = '', show = True):
    
    x,y,z = starting_vals
    xs, ys, zs = [x],[y],[z]
    step, num_steps = .1, 1000
    for t in range(num_steps):
        x,y,z = update_xyz(x,y,z, sigma,rho, beta)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    # Plot
    #print(xs)
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(projection='3d')


    ax.plot(xs, ys, zs, lw=1)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(
                 r'$x_t = \sigma(y-x), y_t = x(\rho-z)-y, z_t = xy-\beta z$'+'\n'+
                 r'$\sigma =$' +f'{sigma:.1f}, '+r'$\rho =$' +f'{rho:.1f}, '+r'$\beta =$' +f'{rho:.1f}'+"\n"+
                 f'Starting Vals = {starting_vals}, Ending Vals = [{x:.1f}, {y:.1f}, {z:.1f}]'+'\n'+
                 f'Step Size = {step}, Total Steps = {num_steps}\n{title}')
    
    if show:
        plt.show()
    return fig, ax


def save_lorenz_video(fname = 'run0_lr0.01_A0_B1_epochs5000_cos_nonorm', savefile='lorenz_attractor.gif'):
    # Set up formatting for the movie files
    dir = 'optimize'
    title = 'lorenz'
    full_file = dir+'/'+title+'_3params'+'/'+fname+'_parameters.npy'
    parameters = np.load(full_file)
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    
    # This list will hold the plots for updating each frame.
    plots = []

    # Prepare the plots for each parameter set.
    for i, param in enumerate(parameters):
        print(i)
        sigma, rho, beta = param
        fig, ax = plot_lorenz_3d(sigma, rho, beta, title=f'Training epoch {10*i}', show = False)
        plots.append([ax])

    # Create the animation object. The blit parameter is a performance optimization.
    anim = animation.ArtistAnimation(fig, plots, interval=200, blit=True, repeat_delay=1000)

    # Save as GIF
    anim.save(savefile, writer='pillow', fps=10)


if __name__ == '__main__':
    create_lorenz_bases(3)
    run_many_lorenz()
    #look_at_many_lorenz()
    #sigma = 10
    #rho = 35
    #beta = 8/3
    #plot_lorenz_3d(sigma, rho, beta, title = 'Training epoch 100')
    #look_at_many_lorenz()
    #save_lorenz_video()
    #fs = ['x_t = {0}*y-{0}*x', 'y_t = {1}*x-x*z-y', 'z_t = x*y-{2}*z']
    
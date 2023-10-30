import numpy as np
from model import find_cq
from util import read_bases
from calculate_G import create_matrices
from contextlib import redirect_stdout
import io
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from util import calculate_neff
from tqdm import tqdm


def evaluate_k(f,k):
    # we plug values of k into f and return the singular values of the solution
    # and the number of conserved quantities
    f = f.format(k=k)
    b = read_bases()
    # Suppress the print statements from find_cq
    with io.StringIO() as buf, redirect_stdout(buf):
        results = find_cq(f,b, check_trivial_bases = True, check_trivial_solutions=False)
    sing = results['s_cq_nonorm'][::-1]
    num_cq = len(results['sol_cq_sparse'])
    bases = results['bases']
    return sing, num_cq, bases

def sweep_k(f, title):
    # we sweep k from -10 to 10 and save the singular values as "{title}_sings.npy"
    # this is useful for plotting what the landscape of n_eff looks like as a function of k
    sings = []
    start, stop = -10, 10
    step = 0.1
    k = start
    while k <= stop:
        print(k)
        sing = evaluate_k(f,k)[0]
        sings.append(sing)
        k+=step
    np.array(sings)
    np.save(f'{title}_sings.npy', sings)


def sweep_kdv(A = 0, B = 10):
    thetas = np.linspace(0, np.pi, 100)
    #phis = np.linspace(0, np.pi, 100)
    phis = np.array([np.pi/2])
    graph_space = []

    total_iterations = len(thetas) * len(phis)
    pbar = tqdm(total=total_iterations, desc="Processing", leave=True)

    for theta in thetas:
        for phi in phis:
            coef1 = np.cos(theta)*np.sin(phi)
            coef2 = np.sin(theta)*np.sin(phi)
            coef3 = np.cos(phi)
            f = f'u_t = {coef1}*u_xxx+{coef2}*u*u_x+{coef3}*u_xx'
            fs = [f]
            bs = read_bases('kdv')
            us = np.load('test_curves.npy')
            G = create_matrices(bs, fs, us)
            _,s,_ = np.linalg.svd(G)
            n_eff = calculate_neff(s, A = A, B = B)
            graph_space.append([theta, phi, n_eff])

            # Update the progress bar
            pbar.set_description(f"f: {f}, n_eff: {n_eff:.5f}")
            pbar.update(1)

    # Close the progress bar
    pbar.close()
    graph_space = np.array(graph_space)
    np.save(f'kdv_coef_space3d_A{A}B{B}.npy', graph_space)

def plot_kdv_coef_space():
    graph_space = np.load('kdv_coef_space.npy')
    diff = 7
    
    # Calculate local maxima
    y_values = graph_space[:,1] - diff
    local_maxima = (y_values[:-2] < y_values[1:-1]) & (y_values[1:-1] > y_values[2:])
    x_local_maxima = graph_space[1:-1, 0][local_maxima]
    y_local_maxima = y_values[1:-1][local_maxima]
    
    plt.plot(graph_space[:,0], y_values, color='b')
    plt.scatter(x_local_maxima, y_local_maxima, color='r', label='Local Maxima\n'+str(x_local_maxima))  # plot local maxima
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$n_{eff}$'+f'-{diff}')
    title = 'u_t = cos(theta)*u_xxx+sin(theta)*u*u_x\nA = 0, B = 10\n'
    title += r'$n_{eff}$ as a function of $\theta$ for KdV'
    plt.title(title)
    
    # Set xticks every pi/4
    ticks = np.arange(0, 2*np.pi, np.pi/4)
    plt.xticks(ticks, [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
                       r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'])
    
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.legend()
    plt.show()

def plot_kdv3d_coef_space():
    graph_space = np.load('kdv_coef_space3d.npy')
    theta, phi, n_eff = graph_space[:,0], graph_space[:,1], graph_space[:,2]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(theta, phi, c=n_eff, cmap='viridis', s=50)

    # Use Greek letters for axis labels
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    
    # Set x and y ticks every pi/4
    ticks = np.arange(0, np.pi+0.1, np.pi/4) 
    tick_labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.set_title(r'$u_t = \cos(\theta)\sin(\phi)u_{xxx}+\sin(\theta)\sin(\phi)uu_x+\cos(\phi)u_{xx}$' + '\nA = 0, B = 10\n' + r'$n_{eff}$ as a function of $\theta$ and $\phi$ for KdV')

    # Add a colorbar to the plot
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('n_eff')

    plt.show()


def plot_A_B(f, title, A = 2, B = 1000):
    # plots the singular values annotated with n_eff for a single A, B
    s = np.load(title+'_sings.npy')
    ks = np.linspace(-10,10,201)
    vals = calculate_neff(s, A = A, B = B)
    plt.plot(ks, vals, c = 'b')
    plt.xlabel('k')
    plt.legend()
    plottitle = r'$(1+e^{-(A-\log(s_i))/B})^{-1}$'+f'\nA = {A}, B = {B}'
    plt.ylabel(plottitle)
    plt.title(f'{f}\n{plottitle}')
    plt.yscale('log')
    plt.show()

def plot_grid_A_B(f, title, A = None, B = None):
    #we plot the n_eff vs k curve for singular values for different values of A and B
    #this is plotted on a grid. we annotate each subplot with the max n_eff value
    s = np.load(f'{title}_sings.npy')
    ks = np.linspace(-10,10,len(s))
    if A is None:
        A = [10,5, 0, -5, -10]
    if B is None:
        B = [1, 10, 100,1000]
    
    fig, axs = plt.subplots(len(A), len(B), figsize=(len(A)*3, len(B)*3))
    
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            vals = calculate_neff(s, A=a, B=b)
            vals = np.array(vals)
            val = np.max(vals)
            
            # Plot vals for the specific a, b
            axs[i, j].plot(ks, vals, color = 'b')
            axs[i, j].set_title(f"A={a}, B={b}")
            
            # Removing ticks, labels, and numbers
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
            # annotate plot with val
            axs[i, j].annotate(f"max neff= {val:.2f}", xy=(0.1, 0.8), xycoords='axes fraction', fontsize=12)
    plt.title(f)
    # Adjusting space between subplots for clarity
    plt.tight_layout()
    plt.show()


def plot_small_k(f, A = 2, B = 1000):
    # we plot the singular value graphs for small values of k in f
    # this is useful for determining the threshold for the singular value and understanding how quickly it changes
    ks = [10**i for i in range(-4, 2)]
    ks[0] = 0
    sings = []
    for k in ks:
        s = evaluate_k(f, k)[0]
        sings.append(s)
    sings = np.array(sings)
    cons = calculate_neff(sings, A, B)
    for i, k in enumerate(ks):
        score = np.sum(cons[i])
        x = list(range(len(sings[i])))
        plt.scatter(x, sings[i], label = f'k = {k}; CQeff = {score:.2f}')
        plt.plot(x, sings[i], lw = 0.1)
    plt.axhline(A, c = 'r', ls = '--', label = 'A (SV Thresh)')
    plt.legend()
    plt.title(f'{f}\nCQeff = '+r'$(1+e^{-(A-\log_{10}(s_i))/B})^{-1}$'+f'\nA = {A}, B = {B}')
    plt.xlabel('CQ')
    plt.ylabel('Score')
    plt.yscale('log')
    plt.show()




if __name__ == '__main__':
    #f, title = 'u_xxx-6*u*u_x+{k}*u_xx', 'kdv'
    #f, title = 'u_x*u+{k}*u_xx', 'burgers'
    #plot_small_k(f,5, 1000)
    sweep_kdv(A = 0, B = 1)
    #plot_kdv3d_coef_space()
    #plot_kdv_coef_space()




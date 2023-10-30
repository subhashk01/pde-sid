from calculate_G_torch import create_matrices_torch
from calculate_G_torch_optimize import generate_fdiffs_bgrads, create_matrices_fast
from util import calculate_neff_torch, read_bases, fill_placeholders
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, re, math

def optimize(f, title,lr = .001, A = 0, B = 1000, epochs = 5000, T_max=None, fname = '', initials = None):
    bs = read_bases(title)
    us = np.load('test_curves.npy')
    

    numbers = [int(match) for s in f for match in re.findall(r'{(\d+)}', s)]

    # Create a list of trainable torch parameters
    #param_list = [torch.nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(max(numbers)+1)]
    #starting_vals = torch.tensor([.4, -.6, .1])
    if initials is not None:
        starting_vals = initials.clone().detach()
    else:
        starting_vals = torch.randn(3)
    # make param_list trainable params
    sum_of_squares = sum(val**2 for val in starting_vals)

    norm = torch.sqrt(sum_of_squares)
    print(starting_vals/norm)

    theta = torch.arccos(starting_vals[-1]/norm)
    phi = torch.arctan(starting_vals[1]/starting_vals[0])
    param_list = [theta, phi]

    # make theta and phi trainable parameters
    param_list = [torch.nn.Parameter(p, requires_grad=True) for p in param_list]

    title = title+f'_{len(param_list)}params' #
    precursor = f'{fname}lr{lr}_A{A}_B{B}_epochs{epochs}_cos_sphereparam'
    print(f'{title}/{precursor}')


    optimizer = torch.optim.Adam(param_list, lr=lr)

    # If T_max isn't provided, set it to epochs by default
    if T_max is None:
        T_max = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


    # Initialize the progress bar
    pbar = tqdm(range(epochs), desc="Optimizing", leave=True)

    loss_list = []
    parameters_list = []


    fdict, bdict = generate_fdiffs_bgrads(f, bs, us, starting_vals) # change starting vals here
    
    for _ in pbar:
        # Forward pass: compute the loss using the current parameters
        # sum_of_squares = sum(p**2 for p in param_list)
        # normalization_factor = torch.sqrt(sum_of_squares)
        # for p in param_list:
        #     p.data /= normalization_factor
        values = [torch.sin(param_list[0])*torch.cos(param_list[1]),
                   torch.sin(param_list[0])*torch.sin(param_list[1]), 
                   torch.cos(param_list[0])]
        #G = create_matrices_torch(bs, f, us, param_list)
        G = create_matrices_fast(f, bs, us, values, bdict, fdict)

        _,s,_ = torch.linalg.svd(G)

        neff = calculate_neff_torch(s, A = A, B = B) 
        loss = -neff
        actual_neff = calculate_neff_torch(s, A = A, B = 0)
        loss_list.append([loss.item(), actual_neff.item()])

        current_lr = optimizer.param_groups[0]['lr']

        parameter = [p.item() for p in param_list]
        parameters_list.append(parameter)

        f_inp = fill_placeholders(f, ["{:.2e}".format(v) for v in values])

        # Update the progress bar with the desired metrics
        pbar.set_postfix({"f": f_inp, "neff": neff.item(),"actual_neff":actual_neff.item(),"lr": "{:.2e}".format(current_lr)})

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()

    losses = np.array(loss_list)

    directory_path = f'optimize/{title}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


    np.save(f'{directory_path}/{precursor}_loss.npy', losses)

    parameters = np.array(parameters_list)
    np.save(f'{directory_path}/{precursor}_parameters.npy', parameters)

    return [p.item() for p in param_list]



##### ANALYSIS #####



def plot_loss_parameters(f, title, filename):
    numbers = [int(match) for s in f for match in re.findall(r'{(\d+)}', s)]
    #title = title+f'_{max(numbers)+1}params'
    title = title+'_2params'

    #losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    parameters = np.load(f'optimize/{title}/{filename}_parameters.npy')

    epochs = list(range(len(losses)))
    smallest_loss_arg = np.argmin(losses[:,0])
    print(smallest_loss_arg)

    plt.figure(figsize = (8,8))
    plt.plot(epochs, losses[:,0],c = 'b', label = '-n_eff')
    plt.plot(epochs, -losses[:,1],c = 'r',linestyle = '--', alpha = 0.5, label = '-n_eff (actual)')
    plt.scatter(epochs[smallest_loss_arg], losses[smallest_loss_arg,0], c = 'r', label = f'Smallest Loss\n-n_eff = {losses[smallest_loss_arg,0]:.6f} at Epoch {epochs[smallest_loss_arg]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss = -n_eff')


    title_str = f'{f}\nSmallest Loss = {losses[smallest_loss_arg,0]:.4f} at Epoch {epochs[smallest_loss_arg]}\n'
    title_str+=f'{title}/{filename}'
    plt.title(title_str)
    plt.legend()
    plt.show()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize = (12,8))
    theta = parameters[:,0] 
    phi = parameters[:,1]
    parameters = np.zeros((parameters.shape[0], parameters.shape[1]+1))
    parameters[:,0] = np.sin(theta)*np.cos(phi)
    parameters[:,1] = np.sin(theta)*np.sin(phi)
    parameters[:,2] = np.cos(theta)

    str_reps = ['sin(theta)*cos(phi)', 'sin(theta)*sin(phi)', 'cos(theta)']

    for i in range(parameters.shape[1]):
        plt.plot(epochs, parameters[:,i], c = colors[i], label = f'Parameter {i} = {str_reps[i]}\nVal at Smallest Loss = {parameters[smallest_loss_arg,i]:.5f}')
        if np.min(parameters[:,i])<0:
            plt.axhline(0, c = 'k')
        plt.xlabel('Epoch')
        plt.ylabel(f'Parameters Through Training')
        #plt.yscale('symlog')


    
        #plt.scatter(epochs[smallest_param_arg], smallest_param[i], c='r', label = f'Smallest Parameter {i+1}\nParameter 1 = {smallest_param[i]:.3e} at Epoch {epochs[smallest_param_arg]}')
    plt.ylim((-1,1))
    smallest_loss_param = parameters[smallest_loss_arg]
    title_str = str(f)
    title_str += f'\nSmallest Loss at Epoch {epochs[smallest_loss_arg]}\n'
    title_str+=f'{title}/{filename}'
    plt.title(title_str)
    plt.legend()
    plt.show()

def returnspherical_coords(parameters):
    theta, phi = parameters[:,0], parameters[:,1]
    x0 = np.sin(theta)*np.cos(phi)
    x1 = np.sin(theta)*np.sin(phi)
    x2 = np.cos(theta)
    return x0,x1,x2

def plot_kdv_solutions_space(filename):
    dir = 'optimize/kdv_2params/'
    best_params_list = None # Initialize an empty list to collect best_param values

    initials = []
    for i in range(100):
        print(i)
        losses = np.load(f'{dir}run{i}_{filename}_loss.npy')
        parameters = np.load(f'{dir}run{i}_{filename}_parameters.npy')
        best_param = parameters[np.argmin(losses[:, 0])]
        all_best_params = parameters[losses[:, 1] == 5]

        if best_params_list is None:
            best_params_list = all_best_params
        else:
            best_params_list = np.concatenate((best_params_list, all_best_params), axis=0)
        
        
        initial = parameters[0]
        initials.append(initial)

        plt.scatter(initial[0], initial[1], color='b')
        plt.scatter(best_param[0], best_param[1], color='r')

        # Draw lines from the initial point to each best point
        plt.plot([initial[0], best_param[0]], [initial[1], best_param[1]], color='black', alpha=0.05)
        
        plt.scatter(all_best_params[:, 0], all_best_params[:, 1], color='r', alpha=0.5)

    # Set the Greek letter labels and adjust x-ticks
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
    
    plt.xticks([i * np.pi / 4 for i in range(5)], 
               ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    plt.yticks([i * np.pi / 4 for i in range(-2, 3)],
               [r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    
    plt.xlim(0, np.pi)
    plt.ylim(-np.pi/2, np.pi/2)

    plt.title(r"$u_t = \sin(\theta)\cos(\phi)u_{xxx} + \sin(\theta)\sin(\phi)uu_x+\cos(\theta)u_{xx}$" + "\nKdV Solutions in Parameter Space\nBlue Starting Point, Red Best Point, Transparent Red Other Solutions")
    plt.show()

    return np.array(best_params_list), np.array(initials) # Return the list of best_param values


def plot_kdv2d_space(params_list):
    x0, x1, x2 = returnspherical_coords(params_list)
    
    kdv = [1,-6]
    kdv_norm = np.sqrt(np.sum(np.array(kdv)**2))
    kdv = np.array(kdv)/kdv_norm
    # Fit a parabola
    coefficients = np.polyfit(x1, x0, 2)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(min(x1), max(x1), 1000)
    y_fit = poly(x_fit)
    
    x = np.linspace(x1.min(), x1.max(), 1000)
    y = np.sqrt(1-x**2)
    
    
    # Create the parabola equation string
    a, b, c = coefficients
    parabola_equation = f'y = {a:.3f}x^2 + {b:.3f}x + {c:.3f}'
    
    plt.figure(figsize=(6,6))
    plt.plot(x,y, color = 'r', label = '{0} = sqrt(1-{1}^2))')
    plt.scatter(x1, x0, color='b', label='theta = pi/2 solutions ({2} = 0)')
    plt.scatter(kdv[1], kdv[0], color = 'r', label = f'KdV Coefficients ({kdv[1]:.3f}, {kdv[0]:.3f})')
    
    # Plot the fitted parabola and its equation in the legend
    #plt.plot(x_fit, y_fit, 'r-', label=parabola_equation)
    
    plt.xlabel('Parameter {1}')
    plt.ylabel('Parameter {0}')
    plt.title('u_t = {0}u_xxx+{1}u*u_x+{2}u_xx\nSolutions in Parameter Space')
    plt.legend()
    plt.show()

    # Return coefficients for further use if needed
    return coefficients




def plot_loss_landscape(f, title, filename):
    numbers = [int(match) for s in f for match in re.findall(r'{(\d+)}', s)]
    title = title + f'_{max(numbers)+1}params'

    losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    
    parameters = np.load(f'optimize/{title}/{filename}_parameters.npy')
    epochs = list(range(len(losses)))

    for i in range(parameters.shape[1]):
        sorted_indices = np.argsort(parameters[:, i])
        sorted_parameters = parameters[sorted_indices, i]
        sorted_losses = losses[sorted_indices]
        sorted_epochs = np.array(epochs)[sorted_indices]

        plt.figure(figsize=(8, 6))
        
        lowest_loss_arg = np.argmin(sorted_losses)
        # Plotting the line
        #plt.plot(sorted_parameters, sorted_losses, c='b', alpha=0.5, lw=0.5)  # with low alpha for visibility
        # Scatter plotting for color scale
        lowest_loss_val = sorted_parameters[lowest_loss_arg]
        scatter = plt.scatter(sorted_parameters, sorted_losses, c=sorted_epochs, label = f'Parameter {i} = {lowest_loss_val:.2f} at Lowest Loss',cmap='plasma', s=10)
        plt.colorbar(scatter, label='Epochs')

        # min_power = np.floor(np.log10(np.min(np.abs(sorted_parameters[sorted_parameters != 0]))))
        # max_power = np.ceil(np.log10(np.max(np.abs(sorted_parameters))))
        
        # ticks = [10**i for i in range(int(min_power), int(max_power)+1)]
        # ticks = [-tick for tick in reversed(ticks)] + ticks
        
        # plt.xticks(ticks)

        plt.title(f"{f}\n{title}/{filename}\nLoss Landscape for Parameter {i+1}")
        plt.xlabel(f"Parameter {i}")
        plt.ylabel("Loss = -n_eff")
        plt.legend()
        #plt.xlim([min(ticks), max(ticks)])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    


if __name__ == '__main__':



    
    f = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx']
    fname = f'lr{0.001}_A{0}_B{1}_epochs{3500}_cos_sphereparam'
    best_param_list, initials = plot_kdv_solutions_space(fname)
    rows_with_negative = np.any(initials < 0, axis=1)
    count = np.sum(rows_with_negative)

    print('num negative',count)
    print('num positive', len(initials)-count)
    plot_kdv2d_space(best_param_list)
    print(best_param_list.shape)
    # title = 'kdv'
    # lr = 1e-3
    # torch.manual_seed(0)
    # for i in range(100):
    #     Bepochs = [[1,3500]]
    #     starting_vals = torch.randn(3)
    #     print(i)
    #     for Bepoch in Bepochs:
    #         A = 0
    #         B, epochs = Bepoch
    #         fname = f'run{i}_'
    #         filename = f'{fname}lr{lr}_A{A}_B{B}_epochs{epochs}_cos_sphereparam'
    #         #plot_loss_parameters(f, title, filename)
    #         optimize(f, title, lr = lr, epochs = epochs, A = A, B = B, fname = fname, initials = starting_vals)
        


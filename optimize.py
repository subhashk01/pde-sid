from calculate_G_torch import create_matrices_torch
from calculate_G_torch_optimize import generate_fdiffs_bgrads, create_matrices_fast
from util import calculate_neff_torch, give_equation,read_bases, fill_placeholders, spherical_transform, inverse_spherical_transform, get_component_map,extract_rhs, split_equation_components
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, re, math
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def param_list_flat(param_list):
    values = []
    for param_tensor in param_list:
        row_vals = inverse_spherical_transform(param_tensor)
        values.append(row_vals)
    values = torch.cat(values)
    return values

def optimize(fs, title, bases = 'kdv',lr = .001, A = 0, B = 1, epochs = 5000, T_max=None, starting_vals = None, fname = '', save = True, sphere = True):
    bs = read_bases(bases)

    us = np.load('test_curves.npy')

    # for f in fs:
    #     numbers = [int(match) for match in re.findall(r'{(\d+)}', f)]
    #     num_params = max(numbers)-min(numbers)+1
    #     total_params += num_params
    #     values = torch.randn(num_params)
    #     #values = torch.tensor([2.,2.])
    #     if sphere:
    #         sum_of_squares = sum(val**2 for val in values)
    #         norm = torch.sqrt(sum_of_squares)
    #         values /= norm
    #         param_tensor = spherical_transform(values)
    #     else:
    #         param_tensor = values
    #     param_tensor = torch.nn.Parameter(param_tensor, requires_grad=True)
    #     param_list.append(param_tensor)
    numbers = [int(match) for match in re.findall(r'{(\d+)}', fs[0])]
    num_params = max(numbers)-min(numbers)+1
    
    if starting_vals is None: 
        starting_vals = torch.randn(num_params)
    assert(len(starting_vals) == num_params)

    if sphere:
        sum_of_squares = sum(val**2 for val in starting_vals)
        norm = torch.sqrt(sum_of_squares)
        starting_vals /= norm
        param_tensor = spherical_transform(starting_vals)
        #param_tensor = torch.tensor([1.63472752, 1.63485858, 0.95531662, 2.35619449])
        end = 'sphereparam'
    else:
        param_tensor = values
        end = 'nonorm'

    param_tensor = torch.nn.Parameter(param_tensor, requires_grad=True)

    title = title+f'_{num_params}params' 

    precursor = f'{fname}lr{lr}_A{A}_B{B}_epochs{epochs}_cos_{end}'
    print(f'{title}/{precursor}')


    optimizer = torch.optim.Adam([param_tensor], lr=lr)

    # If T_max isn't provided, set it to epochs by default
    if T_max is None:
        T_max = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


    # Initialize the progress bar
    pbar = tqdm(range(epochs), desc="Optimizing", leave=True)

    loss_list = []
    parameters_list = []

    # MAKE LIST OF PARAM TENSORS INTO ONE LONG PARAM TENSOR
    # make copies of param tensor as starting vals

    fdict, bdict = generate_fdiffs_bgrads(fs, bs, us, starting_vals) # change starting vals here
    best_loss = float('inf')
    best_param = None
    
    for epoch in pbar:
        # Forward pass: compute the loss using the current parameters
        if sphere:
            values = inverse_spherical_transform(param_tensor)
        else:
            values = param_tensor
        G = create_matrices_fast(fs, bs, us, values, bdict, fdict)

        _,s,_ = torch.linalg.svd(G)

        neff = calculate_neff_torch(s, A = A, B = B) 
        loss = -neff

        if loss < best_loss:
            best_loss = loss
            best_param = values.clone().detach()

        actual_neff = calculate_neff_torch(s, A = A, B = 0)

        current_lr = optimizer.param_groups[0]['lr']


        f_inp = fill_placeholders(fs, ["{:.2e}".format(v) for v in values])

        # Update the progress bar with the desired metrics
        pbar.set_postfix({ "f": f_inp, "neff": neff.item(),"actual_neff":actual_neff.item(),"f": f_inp,  "lr": "{:.2e}".format(current_lr)})

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()

        if epoch % 10 == 0:
            parameters_list.append(values.detach().numpy())
            loss_list.append([loss.item(), actual_neff.item()])

    losses = np.array(loss_list)
    parameters = np.array(parameters_list)

    if save:
        directory_path = f'optimize/{title}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        np.save(f'{directory_path}/{precursor}_loss.npy', losses)
        np.save(f'{directory_path}/{precursor}_parameters.npy', parameters)

    return starting_vals, best_param



##### ANALYSIS #####



def plot_loss_parameters(f, title, filename, sphere = True):
    component_map = get_component_map(f)

    numbers = [int(match) for s in f for match in re.findall(r'{(\d+)}', s)]
    title = title+f'_{max(numbers)+1}params'

    #losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    parameters = np.load(f'optimize/{title}/{filename}_parameters.npy')

    print(parameters[0,:])

    epochs = list([10*i for i in range(len(losses))])
    smallest_loss_arg = np.argmin(losses[:,0])

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

    # if sphere:
    #     values = []
    #     for param in parameters:
    #         tensor_value = inverse_spherical_transform(torch.from_numpy(param))
    #         values.append(tensor_value)

    #     values = torch.stack(values)
    # else:
    values = torch.from_numpy(parameters)


    colormap = plt.cm.viridis  # or any other colormap
    norm = Normalize(vmin=0, vmax=values.shape[1]-1)
    scalar_map = ScalarMappable(norm=norm, cmap=colormap)

    plt.figure(figsize = (12,8))

    print(values.shape[1])
    
    for i in range(values.shape[1]):
        color = scalar_map.to_rgba(i)
        plt.plot(epochs, values[:,i], c = color, label = f'{{{i}}}:{component_map[i]} Val at Smallest Loss = {values[smallest_loss_arg,i]:.5f}')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel(f'Parameters Through Training')
        #plt.yscale('symlog')
        #plt.scatter(epochs[smallest_param_arg], smallest_param[i], c='r', label = f'Smallest Parameter {i+1}\nParameter 1 = {smallest_param[i]:.3e} at Epoch {epochs[smallest_param_arg]}')
    if sphere:
        plt.ylim((-1,1))
    # sort indices by smallest_loss_param
    smallest_params_absneg = -np.abs(np.asarray(values[smallest_loss_arg,:]))
    sorted_indices = np.argsort(smallest_params_absneg)
    for i in sorted_indices:
        print(component_map[i], values[smallest_loss_arg, i])
    title_str = str(f)
    title_str += f'\nSmallest Loss at Epoch {epochs[smallest_loss_arg]}\n'
    title_str+=f'{title}/{filename}'
    plt.title(title_str)
    plt.legend()
    plt.show()





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



    
    #f = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx+{3}*u**2+{4}*u_x**2']
    f = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx']
    b = read_bases('kdv')
    title = 'kdv'
    lr = 1e-3
    torch.manual_seed(0)
    A, B = 0,1
    epochs = 5000
    fname = ''
    filename = f'{fname}lr{lr}_A{A}_B{B}_epochs{epochs}_cos_sphereparam'
    #plot_loss_parameters(f, title, filename, False)
    

    f = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx+{3}*u**2+{4}*u_x**2']
    filename = 'newlr0.001_A0_B1_epochs5000_cos_sphereparam'
    plot_loss_parameters(f, title, filename)
    #optimize(f, title, lr = lr, epochs = epochs, A = A, B = B, fname = 'new')
        


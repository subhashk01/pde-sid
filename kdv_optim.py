from util import create_polynomial_basis, check_function_integral, inverse_spherical_transform, get_component_map, extract_rhs
import numpy as np
from model import threshold_and_format
import matplotlib.pyplot as plt
from optimize import optimize, plot_loss_parameters
import torch, sys
from itertools import cycle
import pandas as pd
from matplotlib import cm
from sklearn.decomposition import PCA
import os
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from scipy.optimize import least_squares

def create_kdv_basis(max_deriv = 2, max_exp = 3):
    # given the max number of derivs (e.g. 2 is u_xx)
    # and the degree of the polynomial, creates polynomial basis of u'
    # eg if u' = u,u_x, u_xx, and max_exp = 3, then the basis has
    # terms like u_x**2u, u_x**3, u**2u_x, u**3, etc.

    # writes non trivial basis terms to kdv_basis.txt
    variables = ['u']
    for i in range(1, max_deriv+1):
        variables.append('u_'+i*'x')
    bases = create_polynomial_basis(variables, max_exp)
    us = np.load('test_curves.npy')
    with open('bases/kdv_basis.txt', 'w') as f:
        for base in bases:
            val = check_function_integral(base, us, ['u'])
            if not abs(val.mean())<1e-3:
                f.write(base + '\n')
            else:
                print(base, val.mean())

def create_messy_kdv(max_deriv=3, max_exp=3):
    # given the max number of derivs (e.g. 2 is u_xx)
    # and the degree of the polynomial, creates polynomial basis of u'
    # this is used to give a "messy" version of kdv. ideally, 
    # our algorithm would be able to extract  the kdv equation from this
    kdv_eq = 'u_t = {0}*u_xxx+{1}*u*u_x'
    variables = ['u']
    for i in range(1, max_deriv + 1):
        variables.append('u_' + i * 'x')
    bases = create_polynomial_basis(variables, max_exp)
    i = 0
    for b in bases:
        if b == 'u_xxx' or b == 'u*u_x' or b == 'u_x*u' or b =='u_x':
            continue
        else:
            # Use double curly braces to include them as literal characters in the string
            kdv_eq += f'+{{{2 + i}}}*{b}'
            i+=1
    return [kdv_eq]


def plot_kdv_solutions_space():

    parameters = get_all_kdv_messy_parameters(dirs = ['kdv_3params_cluster'])
    best_params = parameters[:, 1, :].numpy()
    starting_vals = parameters[:, 0, :].numpy()
    best_params = inverse_spherical_transform(best_params)
    starting_vals = inverse_spherical_transform(starting_vals)
    return
    # for i in range(100):
    #     print(i)
    #     losses = np.load(f'{dir}run{i}_{filename}_loss.npy')
    #     parameters = np.load(f'{dir}run{i}_{filename}_parameters.npy')
    #     best_param = parameters[np.argmin(losses[:, 0])]
    #     all_best_params = parameters[losses[:, 1] == 5]

    #     if best_params_list is None:
    #         best_params_list = all_best_params
    #     else:
    #         best_params_list = np.concatenate((best_params_list, all_best_params), axis=0)
        
        
    #     initial = parameters[0]
    #     initials.append(initial)

    #     plt.scatter(initial[0], initial[1], color='b')
    #     plt.scatter(best_param[0], best_param[1], color='r')

    #     # Draw lines from the initial point to each best point
    #     plt.plot([initial[0], best_param[0]], [initial[1], best_param[1]], color='black', alpha=0.05)
        
    #     plt.scatter(all_best_params[:, 0], all_best_params[:, 1], color='r', alpha=0.5)

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
    # old code to plot run of 100 curves. deprecated
    x0, x1, x2 = inverse_spherical_transform(params_list) #MIGHT FAIL
    
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


def run_many_kdv_messy(title = 'kdv_messy',num_runs = 100, num_epochs = 5000, start_index = 0, num_nodes = 2, seed = 0):
    kdv = create_messy_kdv(3,3)
    #print(kdv)
    kdv = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx']
    vals = []
    num_vars = len(get_component_map(kdv))
    torch.manual_seed(seed)
    starting_vals = torch.randn(num_runs, num_vars)
    starting_vals = starting_vals[start_index:len(starting_vals):num_nodes]

    run_num = start_index
    losses = []
    for starting_val in starting_vals:

        _, best_param, best_loss = optimize(kdv, title=title, bases='kdv', epochs=num_epochs, save=False, starting_vals=starting_val)
        combined = torch.cat((starting_val.unsqueeze(0), best_param.unsqueeze(0)), dim=0)
        print(combined)
        vals.append(combined)
        vals_tensor = torch.stack(vals)
        fname =  f'optimize/{title}_{num_vars}params/{title}_{run_num}run_{num_epochs}epochs_{num_runs}runs'
        torch.save(vals_tensor,f'{fname}.pt')

        losses.append(best_loss.item())
        torch.save(torch.tensor(losses), f'{fname}_loss.pt')
    # read in tensor
    #vals_tensor = torch.load(f'optimize/{title}_{num_vars}params/{title}_{run_num}run_{num_epochs}epochs_{num_runs}runs.pt')

def get_all_kdv_messy_parameters(dirs):
    parameters = None
    losses = None
    basedir = 'optimize'
    for dir in dirs:
        totaldir = f'{basedir}/{dir}'
        for file in os.listdir(totaldir):
            if 'loss.pt' not in file:
                if parameters is None:
                    parameters = torch.load(f'{totaldir}/{file}')
                else:
                    parameters = torch.cat((parameters, torch.load(f'{totaldir}/{file}')), dim = 0)

                if losses is None:
                    losses = torch.load(f'{totaldir}/{file}'[:-3]+'_loss.pt')
                else:
                    losses = torch.cat((losses, torch.load(f'{totaldir}/{file}'[:-3]+'_loss.pt')), dim = 0)
        
    return parameters, losses


def plot_parallel_coordinates():
    # Load parameters
    parameters = get_all_kdv_messy_parameters()
    best_params = parameters[:, 1, :].numpy()
    starting_vals = parameters[:, 0, :].numpy()

    # Assume `create_messy_kdv` and `extract_rhs` are defined elsewhere
    kdv = create_messy_kdv(3, 3)
    component_map = get_component_map(kdv)

    xaxis = list(range(1, best_params.shape[1] + 1))
    
    # Create a colormap
    colormap = cm.viridis
    color_norm = plt.Normalize(0, len(best_params))
    
    # Plot each line with a unique color
    

    for idx, b_param in enumerate(best_params):
        plt.scatter(xaxis, starting_vals[idx], color=colormap(color_norm(idx)), alpha=0.2)
        plt.plot(xaxis, b_param, color=colormap(color_norm(idx)), alpha=0.5)
            

    
    for x in xaxis:
        plt.axvline(x=x, color='black', alpha=0.1)
    
    plt.xticks(xaxis, [f'{component_map[x-1]}' for x in xaxis], rotation=90)
    plt.ylabel('Best Parameter Value at Lowest Loss')
    plt.title(f'Parameter Space Values at Lowest Loss ({best_params.shape[0]} Runs)\nLines are Best Params, Scatter are Starting Params')
    plt.legend()
    plt.show()

def kdv_messy_manyruns_analytics():
    f = create_messy_kdv(3,3)
    component_map = get_component_map(f)
    

    
    # Load parameters
    parameters = get_all_kdv_messy_parameters()

    best_params = parameters[:, 1, :].numpy()

    list_comps = np.asarray([component_map[i][0] for i in range(len(component_map))])
    eqs = threshold_and_format(list_comps, best_params, precision = 2)
    for i,eq in enumerate(eqs):
        print(f'{i+1}. {eq}')
    

    # Compute average absolute value for each parameter
    avg_abs_values = -np.mean(np.abs(best_params), axis=0)
    sorted_indices = np.argsort(avg_abs_values)
    for i in sorted_indices:
        avg = np.mean(np.abs(best_params[:,i]), axis = 0)
        print(component_map[i],f'{avg:.3f}')

def ellipse_parametric(t, params):
    # Unpack parameters: center (x0, y0, z0), axes lengths (a, b), and rotation angles (phi, theta, psi)
    x0, y0, z0, a, b, phi, theta, psi = params

    # Rotation matrices
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0,            0,           1]
    ])

    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0           ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    Rx = np.array([
        [1, 0,            0           ],
        [0, np.cos(psi), -np.sin(psi)],
        [0, np.sin(psi),  np.cos(psi)]
    ])

    # 2D Ellipse equation
    x = a * np.cos(t)
    y = b * np.sin(t)
    z = np.zeros_like(t)  # Z-coordinate is initially zero

    # Combine into a single 2D array (3 rows, N columns)
    points_2d = np.vstack([x, y, z])

    # Apply rotations and translation
    xyz = Rz @ Ry @ Rx @ points_2d + np.array([[x0], [y0], [z0]])  # Ensure the translation is a column vector
    return xyz

def cost_function(params, points, neff):
    # Calculate the sum of squared distances from each point to the ellipse, weighted by neff^2
    residuals = []
    for point, weight in zip(points, neff):
        # Optimize this part to find the closest point on the ellipse
        t = np.linspace(0, 2 * np.pi, 100)
        distances = np.linalg.norm(ellipse_parametric(t, params) - point[:, np.newaxis], axis=0)
        min_distance = np.min(distances)
        weighted_residual = weight**6 * min_distance**2
        residuals.append(weighted_residual)
    return residuals

# Example usage
# neff = ...  # Array of weights, one for each point in best_params
# result = least_squares(cost_function, initial_guess, args=(best_params, neff))

    
def graph_pca_kdv_messy():
    parameters, losses = get_all_kdv_messy_parameters(dirs = ['kdv_33params_10000_5123cluster'])#, 'kdv_33params_500cluster'])
    neff = -losses

    best_params = parameters[:, 1, :].numpy()
    starting_vals = parameters[:, 0, :].numpy()
    # pca best_params and give how much variance is in each component
    component_map = get_component_map(create_messy_kdv(3,3))
    component_list = np.asarray([component_map[i][0] for i in range(len(component_map))])
    pca = PCA(n_components=3)
    pca.fit(best_params)
    print(pca.explained_variance_ratio_)
    # give me the vectors
    vectors = pca.components_
    threshold = 0.1
    components = threshold_and_format(component_list, vectors, precision = 2, threshold = threshold)
    fig = plt.figure(figsize = (14,8))
    # transform orginal data using pca
    transformed_best = pca.transform(best_params)
    pc1, pc2, pc3 = 0, 1, 2 # i and j are the components we want to plot

    
    ax = plt.axes(projection='3d')

    ax.set_xlabel(f'PC{pc1+1} {100*pca.explained_variance_ratio_[pc1]:.1f}% Var')
    ax.set_ylabel(f'PC{pc2+1} {100*pca.explained_variance_ratio_[pc2]:.1f}% Var')
    ax.set_zlabel(f'PC{pc3+1} {100*pca.explained_variance_ratio_[pc3]:.1f}% Var')

    

    sorted_indices = np.argsort(-neff)
    neff = neff[sorted_indices]
    best_params = best_params[sorted_indices]
    parameters = parameters[sorted_indices]
    transformed_best = transformed_best[sorted_indices]
    eqs = threshold_and_format(component_list, parameters[:,1,:], precision = 2, threshold = 5e-2)

    # upb, lowb = 6, 5.5
    # transformed_best = transformed_best[(neff<upb) &  (neff>lowb)]
    # neff = neff[(neff<upb) &  (neff>lowb)]

    #3d scatter plot
    # Choose a colormap
    cmap = plt.cm.viridis

    # Create a ScalarMappable object with the chosen colormap and the loss values as the bounds
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_array(neff)

    # Use the loss values as color codes
    colors = mappable.to_rgba(neff)
    
    
    #ax.scatter(transformed_best[:,pc1], transformed_best[:,pc2], transformed_best[:,pc3], alpha = .5, color = colors, label = 'Best Parameters PCAd')
    # transform starting data using pca
    transformed_start = pca.transform(starting_vals)
    #ax.scatter(transformed_start[:,pc1], transformed_start[:,pc2], transformed_start[:,pc3], color = 'r', alpha = .1, label = 'Starting Parameters PCAd')
    
    # Create a colorbar
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label('neff')

    ax.scatter(transformed_best[:,pc1], transformed_best[:,pc2], transformed_best[:,pc3], alpha = .5, color = colors, label = 'Best Parameters PCAd')
    # for i in range(1000):
    #     print(f'{i+1}. {neff[i]:.2f}: {eqs[i]}')

    # #Initial guess for parameters: center at origin, unit axes, no rotation
    # initial_guess = [0, 0, 0, 1, 1, 0, 0, 0]

    # # Perform the optimization
    # best_ellipse_params = transformed_best[:, :3][neff<7]
    # result = least_squares(cost_function, initial_guess, args=(best_ellipse_params,neff[neff<7]))

    # # Use the optimized parameters
    # params_optimized = result.x
    # print('Opt Params', params_optimized)
    # t_values = np.linspace(0, 2 * np.pi, 1000)
    # ellipse_points = np.array([ellipse_parametric(t, params_optimized) for t in t_values]).T[0]

    # # Plot the ellipse
    # print(ellipse_points.shape)
    # # save ellipse points
    # np.save('ellipse_points.npy', ellipse_points)

    # ellipse_points = np.load('ellipse_points.npy')
    # #ax.plot(ellipse_points[0], ellipse_points[1], ellipse_points[2], color='red', label='Best Fit Ellipse')
    # ellipse_points = np.swapaxes(ellipse_points, 0, 1)
    # #inverse pca transform ellipse points
    # ellipse_points = pca.inverse_transform(ellipse_points)
    # switch dim 0 and 1 for ellipse_points
    # plot ellipse points
    # ax.plot(ellipse_points[:,0], ellipse_points[:,1], ellipse_points[:,2], color='red', label='Best Fit Ellipse')
    
    # eqs = threshold_and_format(component_list, ellipse_points, precision = 2, threshold = 5e-2)
    # # choose only 100 eqs to look at in order
    # indices = np.linspace(0, len(eqs), 100, endpoint = False, dtype = int)

    # def rotate(angle):
    #  ax.view_init(azim=angle)

    # angle = 3
    # ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    # ani.save('kdv_messy.gif', writer=animation.PillowWriter(fps=15))


    
    print(neff.shape)

    plt.show()

    return
    #cut off u_xxx solutions to see structure
    best_params = best_params[neff<7]
    neff = neff[neff<7]
    # redo the pca in 2dimensions
    pca = PCA(n_components=2)
    pca.fit(best_params)
    print(pca.explained_variance_ratio_)
    # give me the vectors
    vectors = pca.components_
    threshold = 0.1
    components = threshold_and_format(component_list, vectors, precision = 2, threshold = threshold)
    fig = plt.figure(figsize = (14,8))
    # transform orginal data using pca
    transformed_best = pca.transform(best_params)
    pc1, pc2 = 0, 1 # i and j are the components we want to plot
    plt.scatter(transformed_best[:,pc1], transformed_best[:,pc2], color = 'b', alpha = .1, label = 'Best Parameters PCAd')
    plt.show()
    return
    plt.scatter(transformed_best[:,pc1], transformed_best[:,pc2], color = 'b', alpha = .1, label = 'Best Parameters PCAd')
    # transform starting data using pca
    transformed_start = pca.transform(starting_vals)
    plt.scatter(transformed_start[:,pc1], transformed_start[:,pc2], color = 'r', alpha = .1, label = 'Starting Parameters PCAd')
    # draw lines between every start point and best param

    
    for i in range(len(transformed_best)):
        plt.plot([transformed_start[i,pc1], transformed_best[i,pc1]], [transformed_start[i,pc2], transformed_best[i,pc2]], color = 'k', alpha = 0.01)
    
    # find transformed best param with lowest y val
    lowest_y_arg = np.argmin(transformed_best[:,pc2])
    lowest_y = np.array([best_params[lowest_y_arg]])
    low_y_comp = threshold_and_format(component_list, lowest_y, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_y, write the string low_y_comp
    plt.annotate(low_y_comp[0], (0.01+transformed_best[lowest_y_arg,pc1], transformed_best[lowest_y_arg,pc2]))
    
    # do the same for highest_y val
    highest_y_arg = np.argmax(transformed_best[:,pc2])
    highest_y = np.array([best_params[highest_y_arg]])
    high_y_comp = threshold_and_format(component_list, highest_y, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_y, write the string low_y_comp
    plt.annotate(high_y_comp[0], (0.01+transformed_best[highest_y_arg,pc1], transformed_best[highest_y_arg,pc2]))

    # do the same for lowest x val
    lowest_x_arg = np.argmin(transformed_best[:,pc1])
    lowest_x = np.array([best_params[lowest_x_arg]])
    low_x_comp = threshold_and_format(component_list, lowest_x, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_x write the string low_x_comp
    plt.annotate(low_x_comp[0], (transformed_best[lowest_x_arg,pc1],transformed_best[lowest_x_arg,pc2]+.05))

    # do the same for highest x val
    highest_x_arg = np.argmax(transformed_best[:,pc1])
    highest_x = np.array([best_params[highest_x_arg]])
    high_x_comp = threshold_and_format(component_list, highest_x, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_y, write the string low_y_comp
    plt.annotate(high_x_comp[0], (-.5+transformed_best[highest_x_arg,pc1], .05+transformed_best[highest_x_arg,pc2]))


    # # Assuming `transformed_best` is your array of points
    # x_values = transformed_best[:, pc1]
    # y_values = transformed_best[:, pc2]

    # # Calculate the distance from the diagonal y = x
    # # For the first diagonal, the distance of the point (x, y) from the line y = x is |y - x| / sqrt(2)
    # distances = np.abs(y_values + x_values) / np.sqrt(2)
    # print(distances)

    # # Define a tolerance for how close points should be to the line y = x to be considered part of the diagonal
    # tolerance = .5 # This is something you might need to adjust

    # # Select points that are within the tolerance
    # diagonal_points = transformed_best[distances < tolerance]

    # eqs = threshold_and_format(component_list, best_params[distances<tolerance], precision = 2, threshold = 1e-1)
    # # for i, eq in enumerate(eqs):
    # #     print(f'{i+1}. {eq}')

    # # Now, perform linear regression on these points
    # # You can use np.polyfit with a 1st-degree polynomial for a linear fit
    # coefficients = np.polyfit(diagonal_points[:, pc1], diagonal_points[:, pc2], 1)

    # # find r^2 of polynpmoal fit
    # residuals = np.polyval(coefficients, diagonal_points[:,pc1]) - diagonal_points[:,pc2]
    # ss_res = np.sum(residuals**2)
    # ss_tot = np.sum((diagonal_points[:,pc2]-np.mean(diagonal_points[:,pc2]))**2)
    # r_squared = 1 - (ss_res/ss_tot)

    # # Create a polynomial from the coefficients
    # polynomial = np.poly1d(coefficients)

    # # Generate a range of x-values from min to max of the diagonal points
    # fit_x_values = np.linspace(diagonal_points[:, pc1].min(), diagonal_points[:, pc1].max(), 100)

    # # Calculate the y-values from the polynomial
    # fit_y_values = polynomial(fit_x_values)

    # # Plot the line of best fit
    # #plt.plot(fit_x_values, fit_y_values, color='b', linewidth=2, label = f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f} | R^2 = {r_squared:.2f}')

    # # make fit_x_values fit_y_values into single array
    # fit_points = np.array([fit_x_values, fit_y_values]).T
    # # transform fit_points back to original space
    # #fit_points = pca.inverse_transform(fit_points)
    # # normalize every row in fit_points
    # #fit_points = fit_points/np.sqrt(np.sum(fit_points**2, axis = 1, keepdims = True))

    # # i want to find the row of fit_points that has the fewest number of values below a threshold
    # # and then print out the equation of that row
    # # first, threshold the fit_points
    # # now, find the number of values below threshold in each row
    # thresh = 5e-2
    # num_above_thresh = np.sum(np.abs(fit_points) > thresh, axis = 1)
    # print(num_above_thresh)
    # min_row = np.argmin(num_above_thresh)

    # # find the row with the fewest number of values below threshold
    # # min_row is when num_above_thresh == 1
    # # print the equation of that row
    # print(np.sum(fit_points[min_row]**2))
    # eq = threshold_and_format(component_list, np.array([fit_points[min_row]]), precision = 2, threshold = thresh)
    # print(eq)
    




    # Now, perform linear regression on these points
    # You can use np.polyfit with a 1st-degree polynomial for a linear fit
    #coefficients = np.polyfit(diagonal_points[:, 0], diagonal_points[:, 1], 1)

    # Create a polynomial from the coefficients
    #polynomial = np.poly1d(coefficients)
    pc1_var, pc2_var = pca.explained_variance_ratio_[pc1], pca.explained_variance_ratio_[pc2]
    plt.xlabel(f'PC{pc1+1}: {components[pc1]}\nThresh at {threshold}. Explained Var = {pc1_var:.3f}')
    plt.ylabel(f'PC{pc2+1}: {components[pc2]}\nThresh at {threshold}. Explained Var = {pc2_var:.3f}')
    plt.title(f'PC{pc1+1},{pc2+1} of KdV w/ {best_params.shape[1]} Parameters ({best_params.shape[0]} Runs, {pc1_var+pc2_var:.2f} Var Accounted For)')
    plt.legend(loc = 5)
    plt.savefig('figures/kdv_messy_pca.png', bbox_inches = 'tight', dpi = 300)
    plt.show()


# my_task_id = int(sys.argv[1])
# num_tasks = int(sys.argv[2])
# run_many_kdv_messy(num_epochs = 100, title = 'kdv', start_index=my_task_id, num_nodes = num_tasks)
def read_cluster_run():
    for i in range(8):
        t = torch.load(f'optimize/kdv_3params_cluster/kdv_{i}run_100epochs_500runs.pt')
        print(t.shape)


def plot_kdv_3param():
    parameters = get_all_kdv_messy_parameters(dirs = ['kdv_3params_cluster'])

    parameters = parameters[:,1,:].numpy()
    sorted_indices = np.argsort(parameters[:,1])
    parameters = parameters[sorted_indices]
    print(parameters)
    # plot 3d scatter of parameters
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(parameters[:,0], parameters[:,1], parameters[:,2])
    ax.set_xlabel('u_xxx')
    ax.set_ylabel('u_x*u')
    ax.set_zlabel('u_xx')
    plt.show()


def test_lrs():
    print('hi')
    # test different learning rates
    parameters, losses = get_all_kdv_messy_parameters(dirs = ['kdv_33params_10000cluster'])
    sorted_indices = np.argsort(-losses)
    parameters = parameters[sorted_indices]
    losses = losses[sorted_indices]
    starting_vals = parameters[:3,0,:]

    lrs = [1e-2, 5e-2, 1e-3]
    kdv = create_messy_kdv(3,3)
    for lr in lrs:
        for i,starting_val in enumerate(starting_vals):
            print(losses[i])
            optimize(kdv, fname = f'test{i}', title='kdvlr', bases='kdv', epochs=20000, early_stop=True, save=True, starting_vals = starting_val, lr = lr)


if __name__ == '__main__':
    #run_many_kdv_messy(title = 'kdv', num_runs=100, num_epochs=10)
    #plot_parallel_coordinates()
    #plot_parallel_coordinates()
    #create_kdv_basis(2,3)
    #kdv_messy_manyruns_analytics()

    #read_cluster_run()
    #graph_pca_kdv_messy()
    test_lrs()


    kdv = create_messy_kdv(3,3)
    component_map = get_component_map(kdv)
    base_components = [component_map[i][0] for i in range(len(component_map))]
    
    file = 'optimize/kdv_33params/lr0.001_A0_B1_epochs20000_cos_sphereparam'
    starting_vals = [.1]+[.5]*(len(base_components)-1)
    starting_vals = torch.tensor(starting_vals)

    parameters, losses = get_all_kdv_messy_parameters(dirs = ['kdv_33params_10000cluster'])
    neff = -losses
    sorted_indices = np.argsort(losses)

    eqs = threshold_and_format(base_components, parameters[:,1,:], precision = 2, threshold = 5e-2)
    # print(eqs)
    # for i, index in enumerate(sorted_indices):
    #     print(f'{i+1}. {neff[index]:.2f}: {eqs[index]}')
    #     if neff[index]<7:
    #         break
    #graph_pca_kdv_messy()

    #optimize(kdv, title='kdv', bases='kdv', fname = 'uxxx', epochs=5000, save=True, starting_vals = starting_vals)
    
    # params = np.load(file+'_parameters.npy')
    # losses = np.load(file+'_loss.npy')
    # neff = -losses[:,0]
    # print(neff.shape)
    # # running stdev of 100 entries of losses
    # running_stdev = []
    # print(neff)
    # for i in range(100, len(neff)):
    #     running_stdev.append(np.std(neff[i-100:i]))
    # running_stdev = np.array(running_stdev)
    # print(running_stdev)
    # epochs = np.linspace(1000, 16760, 1577)
    # plt.plot(epochs, running_stdev, color = 'b')
    # plt.title(f'Running Stdev of neff over 1000 epochs\n{file}')
    # #plt.plot(epochs, neff[100:], color = 'r', label = 'neff')
    # plt.yscale('log')
    # plt.show()
    #eq = threshold_and_format(base_components, end_vals, threshold = 1e-3)
    #print(eq)
    #optimize(kdv, fname = 'nouxxx', title='kdv', bases='kdv', epochs=50000, early_stop=True, save=True, starting_vals = starting_vals)


    #plot_loss_parameters(kdv, 'kdv', 'lr0.001_A0_B1_epochs20000_cos_sphereparam')
    #c = np.load('test_curves.npy')
    #print(c.shape)
    #plot_kdv_3param()
    #plot_kdv_solutions_space()
    
    #kdv = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx']
    

    #plot_loss_parameters(kdv, 'kdv_messy','lr0.001_A0_B1_epochs10000_cos_sphereparam' )

    #kdv_eq_terms = create_polynomial_basis(['u', 'u_x', 'u_xx', 'u_xxx'], 3)
    #print(len(kdv_eq_terms))
    #basis = ['u', 'u**2', 'u**3', 'u**4', 'u**5', 'u**6', 'u_x**2', 'u_x**3', 'u_x**4', 'u_xx**2', 'u_xx**3', 'u_xx**4', 'u_x*u**2', 'u_x**2*u', 'u', 'u_x', 'u_xx', 'u_xxx', 'u_x*u']
    #print(check_trivial(basis,['u']))
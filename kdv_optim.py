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
    for starting_val in starting_vals:

        _, best_param = optimize(kdv, title=title, bases='kdv', epochs=num_epochs, save=False, starting_vals=starting_val)
        combined = torch.cat((starting_val.unsqueeze(0), best_param.unsqueeze(0)), dim=0)
        print(combined)
        vals.append(combined)
        vals_tensor = torch.stack(vals)
        torch.save(vals_tensor, f'optimize/{title}_{num_vars}params/{title}_{run_num}run_{num_epochs}epochs_{num_runs}runs.pt')
    # read in tensor
    #vals_tensor = torch.load(f'optimize/{title}_{num_vars}params/{title}_{run_num}run_{num_epochs}epochs_{num_runs}runs.pt')

def get_all_kdv_messy_parameters():
    parameters = None
    for i in [5,9,14,18,100]:
        filename = f'kdv_messy_5000epochs_{i}runs.pt'
        if parameters is None:
            parameters = torch.load(f'optimize/kdv_messy_33params/{filename}')
        else:
            parameters = torch.cat((parameters, torch.load(f'optimize/kdv_messy_33params/{filename}')), dim = 0)
    return parameters


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
    
def graph_pca_kdv_messy():
    parameters = get_all_kdv_messy_parameters()
    best_params = parameters[:, 1, :].numpy()
    starting_vals = parameters[:, 0, :].numpy()
    # pca best_params and give how much variance is in each component
    component_map = get_component_map(create_messy_kdv(3,3))
    component_list = np.asarray([component_map[i][0] for i in range(len(component_map))])
    pca = PCA(n_components=2)
    pca.fit(best_params)
    print(pca.explained_variance_ratio_)
    # give me the vectors
    vectors = pca.components_
    threshold = 0.1
    components = threshold_and_format(component_list, vectors, precision = 2, threshold = threshold)
    plt.figure(figsize = (14,8))
    # transform orginal data using pca
    transformed_best = pca.transform(best_params)
    plt.scatter(transformed_best[:,0], transformed_best[:,1], color = 'b', label = 'Best Parameters PCAd')
    # transform starting data using pca
    transformed_start = pca.transform(starting_vals)
    plt.scatter(transformed_start[:,0], transformed_start[:,1], color = 'r', label = 'Starting Parameters PCAd')
    # draw lines between every start point and best param
    for i in range(len(transformed_best)):
        plt.plot([transformed_start[i,0], transformed_best[i,0]], [transformed_start[i,1], transformed_best[i,1]], color = 'k', alpha = 0.05)
    
    # find transformed best param with lowest y val
    lowest_y_arg = np.argmin(transformed_best[:,1])
    lowest_y = np.array([best_params[lowest_y_arg]])
    low_y_comp = threshold_and_format(component_list, lowest_y, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_y, write the string low_y_comp
    plt.annotate(low_y_comp[0], (0.01+transformed_best[lowest_y_arg,0], transformed_best[lowest_y_arg,1]))
    
    # do the same for highest_y val
    highest_y_arg = np.argmax(transformed_best[:,1])
    highest_y = np.array([best_params[highest_y_arg]])
    high_y_comp = threshold_and_format(component_list, highest_y, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_y, write the string low_y_comp
    plt.annotate(high_y_comp[0], (0.01+transformed_best[highest_y_arg,0], transformed_best[highest_y_arg,1]))

    # do the same for lowest x val
    lowest_x_arg = np.argmin(transformed_best[:,0])
    lowest_x = np.array([best_params[lowest_x_arg]])
    low_x_comp = threshold_and_format(component_list, lowest_x, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_x write the string low_x_comp
    plt.annotate(low_x_comp[0], (transformed_best[lowest_x_arg,0],transformed_best[lowest_x_arg,1]+.05))

    # do the same for highest x val
    highest_x_arg = np.argmax(transformed_best[:,0])
    highest_x = np.array([best_params[highest_x_arg]])
    high_x_comp = threshold_and_format(component_list, highest_x, precision = 2, threshold = 5e-2)
    # at the coordinates of lowest_y, write the string low_y_comp
    plt.annotate(high_x_comp[0], (-.5+transformed_best[highest_x_arg,0], .05+transformed_best[highest_x_arg,1]))


    # Assuming `transformed_best` is your array of points
    x_values = transformed_best[:, 0]
    y_values = transformed_best[:, 1]

    # Calculate the distance from the diagonal y = x
    # For the first diagonal, the distance of the point (x, y) from the line y = x is |y - x| / sqrt(2)
    distances = np.abs(y_values + x_values) / np.sqrt(2)
    print(distances)

    # Define a tolerance for how close points should be to the line y = x to be considered part of the diagonal
    tolerance = .5 # This is something you might need to adjust

    # Select points that are within the tolerance
    diagonal_points = transformed_best[distances < tolerance]

    eqs = threshold_and_format(component_list, best_params[distances<tolerance], precision = 2, threshold = 1e-1)
    for i, eq in enumerate(eqs):
        print(f'{i+1}. {eq}')

    # Now, perform linear regression on these points
    # You can use np.polyfit with a 1st-degree polynomial for a linear fit
    coefficients = np.polyfit(diagonal_points[:, 0], diagonal_points[:, 1], 1)

    # find r^2 of polynpmoal fit
    residuals = np.polyval(coefficients, diagonal_points[:,0]) - diagonal_points[:,1]
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((diagonal_points[:,1]-np.mean(diagonal_points[:,1]))**2)
    r_squared = 1 - (ss_res/ss_tot)

    # Create a polynomial from the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate a range of x-values from min to max of the diagonal points
    fit_x_values = np.linspace(diagonal_points[:, 0].min(), diagonal_points[:, 0].max(), 100)

    # Calculate the y-values from the polynomial
    fit_y_values = polynomial(fit_x_values)

    # Plot the line of best fit
    #plt.plot(fit_x_values, fit_y_values, color='b', linewidth=2, label = f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f} | R^2 = {r_squared:.2f}')

    # make fit_x_values fit_y_values into single array
    fit_points = np.array([fit_x_values, fit_y_values]).T
    # transform fit_points back to original space
    fit_points = pca.inverse_transform(fit_points)
    # normalize every row in fit_points
    fit_points = fit_points/np.sqrt(np.sum(fit_points**2, axis = 1, keepdims = True))

    # i want to find the row of fit_points that has the fewest number of values below a threshold
    # and then print out the equation of that row
    # first, threshold the fit_points
    # now, find the number of values below threshold in each row
    thresh = 5e-2
    num_above_thresh = np.sum(np.abs(fit_points) > thresh, axis = 1)
    print(num_above_thresh)
    min_row = np.argmin(num_above_thresh)

    # find the row with the fewest number of values below threshold
    # min_row is when num_above_thresh == 1
    # print the equation of that row
    print(np.sum(fit_points[min_row]**2))
    eq = threshold_and_format(component_list, np.array([fit_points[min_row]]), precision = 2, threshold = thresh)
    print(eq)
    




    # Now, perform linear regression on these points
    # You can use np.polyfit with a 1st-degree polynomial for a linear fit
    #coefficients = np.polyfit(diagonal_points[:, 0], diagonal_points[:, 1], 1)

    # Create a polynomial from the coefficients
    #polynomial = np.poly1d(coefficients)
    
    plt.xlabel(components[0]+f'\nThresh at {threshold}. Explained Var = {pca.explained_variance_ratio_[0]:.3f}')
    plt.ylabel(components[1]+ f'\nThresh at {threshold}. Explained Var = {pca.explained_variance_ratio_[1]:.3f}')
    plt.title(f'PCA of KdV w/ {best_params.shape[1]} Parameters ({best_params.shape[0]} Runs, {np.sum(pca.explained_variance_ratio_):.2f} Var Accounted For)')
    plt.legend(loc = 5)
    plt.savefig('figures/kdv_messy_pca.png', bbox_inches = 'tight', dpi = 300)
    plt.show()

if __name__ == '__main__':
    #run_many_kdv_messy(num_runs=100, num_epochs=5000)
    #plot_parallel_coordinates()
    #plot_parallel_coordinates()
    #create_kdv_basis(2,3)
    #kdv_messy_manyruns_analytics()
    run_many_kdv_messy(num_epochs = 100, title = 'kdv')
    
    #kdv = ['u_t = {0}*u_xxx+{1}*u*u_x+{2}*u_xx']
    

    #plot_loss_parameters(kdv, 'kdv_messy','lr0.001_A0_B1_epochs10000_cos_sphereparam' )

    #kdv_eq_terms = create_polynomial_basis(['u', 'u_x', 'u_xx', 'u_xxx'], 3)
    #print(len(kdv_eq_terms))
    #basis = ['u', 'u**2', 'u**3', 'u**4', 'u**5', 'u**6', 'u_x**2', 'u_x**3', 'u_x**4', 'u_xx**2', 'u_xx**3', 'u_xx**4', 'u_x*u**2', 'u_x**2*u', 'u', 'u_x', 'u_xx', 'u_xxx', 'u_x*u']
    #print(check_trivial(basis,['u']))
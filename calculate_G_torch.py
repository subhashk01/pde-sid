import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from calculate_G import differentiate_f, grad_basis_function, create_matrix, calculate_matrix_columns_presum, calculate_matrix_columns
from util import read_bases, find_highest_derivative, calculate_neff_torch, calculate_neff
import numpy as np
from model import svd_and_sparsify, threshold_and_format
import re

torch.set_default_dtype(torch.float64)



def split_equation_components(s):
    # Split the string by addition or subtraction. 
    # The minus sign will stay with the component after it.
    components = [x.strip() for x in re.split('(\+|\-)', s) if x.strip()]
    
    # If a component is just '+' or '-', then merge it with the next component
    i = 0
    while i < len(components) - 1:
        if components[i] in ['+', '-']:
            # Merge with the next component
            components[i] += components[i+1]
            # Remove the next component
            components.pop(i+1)
        else:
            i += 1
    
    # Remove '+' signs from the start of the components
    components = [comp if not comp.startswith('+') else comp[1:] for comp in components]
    
    # Placeholder list
    placeholder_indices = []
    
    # Iterate over the components and check for placeholders
    for i, comp in enumerate(components):
        if '{}' in comp:
            placeholder_indices.append(i)
            components[i] = comp.replace('{}*', '')  # Remove the placeholder
            
    return components, placeholder_indices



def f_component_diffs(components,num_derivs, us):
    # calculates the derivative matrix for each of the additive components of f
    # returns list of each of the components' derivative matrices
    f_diffs = []
    param_num = 0
    for i,comp in enumerate(components):
        f_diff = differentiate_f(comp, num_derivs, us)
        f_diffs.append(f_diff)
        assert(f_diff.shape == f_diffs[0].shape) # each component f_diff needs to be the same size
    return f_diffs
    
def evaluate_f_diff_components(f_diffs,param_list,place_holder_indices):
    # given the f derivative matrices for each of the components
    # a list of hte parameters that need to be mulitpled against each component
    # and the indices where the parameters are supposed to be placed
    # returns a multiplicative sum to get the full f derivative matrix
    f_diff_total = torch.zeros(f_diffs[0].shape)
    for i,f_diff in enumerate(f_diffs):
        coef = 1
        if i in place_holder_indices:
            coef = i
        f_diff_total += coef*torch.from_numpy(f_diff)
    return f_diff_total


def evaluate_f_diff(f,us, param_list):
    # gets the total f_diff matrix with a certain list of parameters

    # evaluates as many derivs as we have data for
    num_derivs = us.shape[1]-find_highest_derivative(f)-1
    components, place_holder_indices = split_equation_components(f)
    assert(len(param_list) == len(place_holder_indices)) # need as many params as unknowns
    f_diffs = f_component_diffs(components,num_derivs, us)
    f_diff_total = evaluate_f_diff_components(f_diffs, param_list, place_holder_indices)

    return f_diff_total

def create_matrix_column_presum_torch(us,b,f_diff):
    # calculates column presum for each basis
    b_grad = grad_basis_function(b, us)
    b_grad = torch.from_numpy(b_grad)
        
    f_diff_indexed = f_diff[:,:b_grad.shape[1],:]
    column_presum = torch.einsum('ijk,ijk->ik', b_grad, f_diff_indexed)
        
    assert(column_presum.shape == (us.shape[0], us.shape[2])) # should be P, N_p
    return column_presum

def create_matrix_column_torch(us, b, f_diff):
    # calculates the column in G for b

    column_presum = create_matrix_column_presum_torch(us,b,f_diff)
    
    column = torch.sum(column_presum, dim=1)
    assert(column.shape == (us.shape[0],)) # should be P
    return column


def create_matrix_torch(bs, f, us, param_list):
    # creates the whole matrix for G. Size P x num_basis

    f_diff = evaluate_f_diff(f, us, param_list)
    matrix = torch.zeros(us.shape[0], len(bs))
    for i,b in enumerate(bs):
        column = create_matrix_column_torch(us, b, f_diff)
        matrix[:,i] = column
    return matrix


###############################################
#  CHECK IF NUMPY MATRICES ARE SAME AS TORCH  #
###############################################


def within_tolerance(tensor1, tensor2, percentage=.1):
    # Calculate the relative difference
    relative_diff = torch.abs(tensor1 - tensor2) / torch.clamp(torch.max(torch.abs(tensor1), torch.abs(tensor2)), min=1e-10)
    
    # Find where the relative difference exceeds the threshold
    mask = relative_diff >= (percentage / 100.0)
    
    # If any exceed the threshold, print out the indices and values
    if torch.any(mask):
        indices = torch.nonzero(mask).tolist()
        for idx in indices:
            print(f"At index {idx}, tensor1 has value {tensor1[tuple(idx)]} and tensor2 has value {tensor2[tuple(idx)]}.")
    
    # Return if all values are within tolerance
    return not torch.any(mask)

def check_torch_implementation(bs, f, us, param_list):
    f_inp = f.format(*param_list)

    num_derivs = us.shape[1]-find_highest_derivative(f)-1
    deriv_f_torch = evaluate_f_diff(f, us, param_list)
    deriv_f_numpy = torch.from_numpy(differentiate_f(f_inp, num_derivs, us))

    assert within_tolerance(deriv_f_torch, deriv_f_numpy), "f_derivatives are not the same."
    print('f_derivs calculated correctly')  

    for b in bs:
        print(b)
        presum_numpy = torch.from_numpy(calculate_matrix_columns_presum(b,f_inp, us))
        presum_torch = create_matrix_column_presum_torch(us, b, deriv_f_torch)
        assert within_tolerance(presum_numpy, presum_torch), "column_presums are not the same"
        print('column presums calculated correctly')

        col_numpy = torch.from_numpy(calculate_matrix_columns(b, f_inp, us)).double()
        col_torch = create_matrix_column_torch(us, b, deriv_f_torch).double()

        col_same = within_tolerance(col_numpy, col_torch)
        #assert col_same, "columns are not the same"
        print('columns calculated correctly')

def compare_answers(bs,f,us,param_list):
    f_inp = f.format(*param_list)
    G_np = create_matrix(bs, f_inp, us)
    res_np = svd_and_sparsify(G_np)

    G_tor = create_matrix_torch(bs, f, us, param_list)
    res_tor = svd_and_sparsify(np.asarray(G_tor))

    print(f, param_list)
    print('\n Numpy Implementation')
    print('Singular Values (norm)')
    print(res_np['s_cq'])
    ans_np = threshold_and_format(bs,res_np['sol_cq_sparse'])
    print(ans_np)
    n_eff = calculate_neff([res_np['s_cq_nonorm']])
    print(f'n_eff:{n_eff}')

    print('\n Torch Implementation')
    print('Singular Values (norm)')
    print('tor', res_tor['s_cq'])
    ans_tor = threshold_and_format(bs,res_tor['sol_cq_sparse'])
    n_eff = calculate_neff_torch(torch.from_numpy(res_tor['s_cq_nonorm']))
    print(ans_tor)
    print(f'n_eff:{n_eff}')




def check_torch():
    fs = ['u*u_x','u_xxx-6*u*u_x+{}*u_xx', '{}*u_xxx-12*u_xx+{}*u_x**3']
    bs = read_bases()
    us = np.load('test_curves.npy')
    for f in fs:
        _, placeholder = split_equation_components(f)
        param_list = [np.random.randint(10) for _ in range(len(placeholder))]
        print(f,param_list)
        #check_torch_implementation(f,bs,us,param_list)
        compare_answers(bs, f, us, param_list)
        
        





if __name__ == '__main__':
    check_torch()





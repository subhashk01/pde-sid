import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from calculate_G import differentiate_f, grad_basis_function, create_matrices, calculate_matrix_columns_presum, calculate_matrix_columns, create_matrix
from util import read_bases, give_equation,find_highest_derivative, calculate_neff_torch, calculate_neff, extract_lhs_variables, extract_rhs, extract_variables, fill_placeholders
import numpy as np
from model import svd_and_sparsify, threshold_and_format
import re
from contextlib import redirect_stdout
import io
from calculate_G_torch import split_equation_components

torch.set_default_dtype(torch.float64)






def f_component_diffs(components,num_derivs, us, variables):
    # calculates the derivative matrix for each of the additive components of f
    # returns list of each of the components' derivative matrices
    f_diffs = []
    for i,comp in enumerate(components):
        f_diff = differentiate_f(comp, num_derivs, us, variables)
        f_diffs.append(f_diff)
        assert(f_diff.shape == f_diffs[0].shape) # each component f_diff needs to be the same size
    return f_diffs
    
def evaluate_f_diff_components(f_diffs,param_list,place_holder_indices, placeholder_nums):
    # given the f derivative matrices for each of the components
    # a list of hte parameters that need to be mulitpled against each component
    # and the indices where the parameters are supposed to be placed
    # returns a multiplicative sum to get the full f derivative matrix
    f_diff_total = torch.zeros(f_diffs[0].shape)
    j = 0 # index of param_list we're on
    for i,f_diff in enumerate(f_diffs):
        coef = 1
        if i in place_holder_indices:
            coef = param_list[placeholder_nums[j]]
            j+=1
        f_diff_total += coef*torch.from_numpy(f_diff)
    return f_diff_total


def evaluate_f_diff(f,us, param_list, variables):
    # gets the total f_diff matrix with a certain list of parameters
    # evaluates as many derivs as we have data for
    num_derivs_poss = []
    for var in variables:
        num_derivs_poss.append(find_highest_derivative(f, wrt=var))

    num_derivs = us.shape[2]-max(num_derivs_poss)-1
    components, place_holder_indices, placeholder_nums = split_equation_components(f)
    
    assert(len(param_list) == len(place_holder_indices)) # need as many params as unknowns
    if len(placeholder_nums):
        assert(max(placeholder_nums) < len(param_list)) # placeholder nums must be in range of param_list
    f_diffs = f_component_diffs(components,num_derivs, us, variables)
    f_diff_total = evaluate_f_diff_components(f_diffs, param_list, place_holder_indices, placeholder_nums)

    return f_diff_total

def create_matrix_column_presum_torch(us,b,f_diff, wrt, variables):
    # calculates column presum for each basis
    b_grad = grad_basis_function(b, us, wrt, variables)
    b_grad = torch.from_numpy(b_grad)
        
    f_diff_indexed = f_diff[:,:b_grad.shape[1],:]
    column_presum = torch.einsum('ijk,ijk->ik', b_grad, f_diff_indexed)
        
    assert(column_presum.shape == (us.shape[1], us.shape[3])) # should be P, N_p
    return column_presum

def create_matrix_column_torch(us, b, f_diff, wrt, variables):
    # calculates the column in G for b

    column_presum = create_matrix_column_presum_torch(us,b,f_diff, wrt, variables)
    
    column = torch.sum(column_presum, dim=1)
    assert(column.shape == (us.shape[1],)) # should be P
    return column



def create_matrix_torch(bs, f, us, wrt, variables, param_list):
    # creates the whole matrix for G. Size P x num_basiS. for each f
    f_diff = evaluate_f_diff(f, us, param_list, variables)
    matrix = torch.zeros(us.shape[1], len(bs))
    for i,b in enumerate(bs):
        column = create_matrix_column_torch(us, b, f_diff, wrt, variables)
        matrix[:,i] = column
    return matrix

def create_matrices_torch(bs, fs, us, param_list):
    assert type(fs)==list, "fs must be a list of string equations"
    matrix = None
    variables = extract_lhs_variables(fs)
    all_vars = extract_variables(fs)
    assert set(variables) == set(all_vars), "# equations must be # total variables"
    equations = extract_rhs(fs) # we dont want the X_t = part

    matrix = None
    for i in range(len(equations)):
        matrix_interim = create_matrix_torch(bs, equations[i], us, variables[i], variables, param_list)
        if matrix is None:
            matrix = torch.zeros(matrix_interim.shape)
        matrix+=matrix_interim
    assert(matrix.shape == (us.shape[1], len(bs))) # should be P, num_bs
    return matrix



def create_matrices_fast(b_grads, f_grads, param_list):

    pass




if __name__ == '__main__':
    # NO SPACES AFTER *
    # fs = ['u_t = v', 'v_t = u']
    # bs = read_bases('kdv')
    # us = np.load('test_curves.npy')
    # param_list = []
    # mat = create_matrices_torch(bs, fs, us, param_list)
    # u,s,v = torch.linalg.svd(mat)
    # print(s)
    # print(split_equation_components('u_xxx - 6*u*u_x+{0}*u_xx'))
    f = ['{0}*v', '-u']
    variables = ['u','v']
    us = np.load('test_curves.npy')
    param_list = [1]
    fdiff = evaluate_f_diff(f[0], us, param_list, variables)
    print(fdiff.shape)




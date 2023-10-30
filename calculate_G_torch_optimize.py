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
from calculate_G_torch import split_equation_components, f_component_diffs, evaluate_f_diff_components, create_matrices_torch

torch.set_default_dtype(torch.float64)




def evaluate_f_diff(f,param_list, f_diffs):
    # gets the total f_diff matrix with a certain list of parameters
    # evaluates as many derivs as we have data for
    _, place_holder_indices, placeholder_nums = split_equation_components(f)
    
    assert(len(param_list) >= len(place_holder_indices)) # need at least as many params as unknowns
    if len(placeholder_nums):
        assert(max(placeholder_nums) < len(param_list)) # placeholder nums must be in range of param_list

    f_diff_total = evaluate_f_diff_components(f_diffs, param_list, place_holder_indices, placeholder_nums)

    return f_diff_total

def create_matrix_column_torch(us,b_grad,f_diff_total):
    # for every b grad taken w.r.t a variable, calculates the presum value 
        
    f_diff_indexed = f_diff_total[:,:b_grad.shape[1],:]
    column_presum = torch.einsum('ijk,ijk->ik', b_grad, f_diff_indexed)
        
    assert(column_presum.shape == (us.shape[1], us.shape[3])) # should be P, N_p
    column = torch.sum(column_presum, dim=1)
    assert(column.shape == (us.shape[1],)) # should be P
    return column



# idea: i give you f_diffs and b_grads ahead of time 

def create_matrices_fast(equations, bs, us, param_list, b_grads_dict, f_grads_dict):
    fs = extract_rhs(equations)
    variables = extract_lhs_variables(equations)
    matrix = torch.zeros(us.shape[1], len(bs))
    for i,f in enumerate(fs):
        f_diff = f_grads_dict[f]
        f_diff_total = evaluate_f_diff(f, param_list, f_diff)
        for j, b in enumerate(bs):
            wrt = variables[i]
            column = create_matrix_column_torch(us, b_grads_dict[b][wrt], f_diff_total)
            matrix[:,j] += column


    return matrix

def generate_fdiffs_bgrads(equations, bs, us, param_list):
    # gets the total f_diff matrix with a certain list of parameters
    # evaluates as many derivs as we have data for
    fs = extract_rhs(equations)
    variables = extract_lhs_variables(equations)

    f_diffs_dict = {}

    for f in fs:
        num_derivs_poss = []
        for var in variables:
            num_derivs_poss.append(find_highest_derivative(f, wrt=var))

        num_derivs = us.shape[2]-max(num_derivs_poss)-1
        components, place_holder_indices, placeholder_nums = split_equation_components(f)
        
        assert(len(param_list) >= len(place_holder_indices)) # need as many params as unknowns
        if len(placeholder_nums):
            assert(max(placeholder_nums) < len(param_list)) # placeholder nums must be in range of param_list
        f_diffs = f_component_diffs(components,num_derivs, us, variables)

        f_diffs_dict[f] = f_diffs

    b_grads_dict = {}
    for b in bs:
        b_grads = {}
        for var in variables:
            b_grads[var] = torch.from_numpy(grad_basis_function(b, us, var, variables))
        b_grads_dict[b] = b_grads
    return f_diffs_dict, b_grads_dict
    


if __name__ == '__main__':
    # NO SPACES AFTER *

    equations = give_equation('nlse')
    print(equations)
    param_list = []
    us = np.load('test_curves.npy')
    bs = read_bases('nlse')
    fdict, bdict = generate_fdiffs_bgrads(equations, bs, us, param_list)
    matrix_fast = create_matrices_fast(equations, bs, us, param_list, bdict, fdict)
    print('fast')
    _,s,_ = torch.linalg.svd(matrix_fast)
    print(s)

    matrix_old = create_matrices_torch(bs, equations, us, param_list)
    print('old')
    _,s,_ = torch.linalg.svd(matrix_old)
    print(s)
    




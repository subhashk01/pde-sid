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

torch.set_default_dtype(torch.float64)



def split_equation_components(s):
    # Split the string by addition or subtraction. 
    # The minus sign will stay with the component after it.

    # doesn't work great w/ f strings
    # e.g. f'0*u_x+{coef}*u_xx' might not work (brackets probably confuse it)
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
    placeholder_nums = []

    # Iterate over the components and check for placeholders
    for i, comp in enumerate(components):
        matches = re.findall(r'{(\d+)}', comp) # Extract numbers inside {}
        
        if matches:
            placeholder_indices.append(i)
            for match in matches:
                placeholder_nums.append(int(match))
                comp = comp.replace('{' + match + '}*', '')  # Remove the placeholder with its number

        components[i] = comp
            
    return components, placeholder_indices, placeholder_nums



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
    # creates the whole matrix for G. Size P x num_basiS
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

    num_derivs = us.shape[2]-find_highest_derivative(f)-1
    deriv_f_torch = evaluate_f_diff(f, us, param_list)
    deriv_f_numpy = torch.from_numpy(differentiate_f(f_inp, num_derivs, us))

    assert within_tolerance(deriv_f_torch, deriv_f_numpy), "f_derivatives are not the same."
    print('f_derivs calculated correctly')  

    # for b in bs:
    #     print(b)
    #     presum_numpy = torch.from_numpy(calculate_matrix_columns_presum(b,f_inp, us))
    #     presum_torch = create_matrix_column_presum_torch(us, b, deriv_f_torch)
    #     assert within_tolerance(presum_numpy, presum_torch), "column_presums are not the same"
    #     print('column presums calculated correctly')

    #     col_numpy = torch.from_numpy(calculate_matrix_columns(b, f_inp, us)).double()
    #     col_torch = create_matrix_column_torch(us, b, deriv_f_torch).double()

    #     col_same = within_tolerance(col_numpy, col_torch)
    #     #assert col_same, "columns are not the same"
    #     print('columns calculated correctly')

    G_np = torch.from_numpy(create_matrix(bs,f_inp,us)).double()
    G_tor = create_matrix_torch(bs,f, us, param_list).double()
    _, s_np, _ =  torch.linalg.svd(G_np)
    _, s_tor, _ = torch.linalg.svd(G_tor)
    print('Numpy Singular Values', s_np)
    print('Torch Singular Values', s_tor)


    matrices_same = within_tolerance(G_np, G_tor, percentage = 10)
    #assert matrices_same, "G matrices are not the same"

    sing_values_same = within_tolerance(s_np, s_tor)
    #assert sing_values_same, "Singular Values are not the same"





def compare_answers(bs,f,us,param_list):
    f_inp = fill_placeholders(f, param_list)
    print(f_inp)
    G_np = create_matrices(bs, f_inp, us)
    G_tor = create_matrices_torch(bs, f, us, param_list)

    with io.StringIO() as buf, redirect_stdout(buf):
        res_np = svd_and_sparsify(G_np)
        res_tor = svd_and_sparsify(np.asarray(G_tor))

    print(f'\n{f_inp}')
    ans_np = res_np['sol_cq_sparse']
    
    if ans_np.shape[1]!=0:
        ans_np = np.array(threshold_and_format(bs,ans_np))
    ans_np = np.ndarray.tolist(ans_np)
    print('NUMPY SOLUTIONS',ans_np)
    n_eff_np = calculate_neff([res_np['s_cq_nonorm']])[0]
    
    

    ans_tor = res_tor['sol_cq_sparse']
    if ans_tor.shape[1]!=0:
        ans_tor = np.array(threshold_and_format(bs,ans_tor))
    ans_tor = np.ndarray.tolist(ans_tor)
    print('TORCH SOLUTIONS',ans_tor)
    n_eff_tor = calculate_neff_torch(torch.from_numpy(res_tor['s_cq_nonorm'])).item()

    print('np sing values',res_np['s_cq_nonorm'])
    print('tor sing values', res_tor['s_cq_nonorm'])
    #assert np.allclose(res_np['s_cq_nonorm'], res_tor['s_cq_nonorm'], rtol=1e-4, atol=0), "Singular values don't match"
    print(f'neff numpy: {n_eff_np:.2f}. neff torch: {n_eff_tor:.2f}. frac diff: {abs(n_eff_tor-n_eff_np)/n_eff_np:.2e}')

    #assert ans_tor == ans_np, f"Numpy and Torch solutions don't match for {f_inp},\nNumpy:{ans_np}\nTorch:{ans_tor}"
    assert abs(n_eff_np)*1.01>abs(n_eff_tor)>abs(n_eff_np)*.99, f"Numpy and Torch n_eff answers don't match for {f_inp}.\nNumpy:{n_eff_np}\nTorch:{n_eff_tor}"




def check_torch():
    # fix this first case
    eqs = ['nlse', 'spring']
    us = np.load('test_curves.npy')
    for eq in eqs:
        f = give_equation(eq)
        bs = read_bases(eq)
        numbers = [int(match) for s in f for match in re.findall(r'{(\d+)}', s)]
        if len(numbers) == 0:
            param_list = []
        else:
            param_list = [np.random.randint(10) for _ in range(max(numbers)+1)]
        #check_torch_implementation(bs,f,us,param_list)
        compare_answers(bs,f,us,param_list)

        
        





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
    check_torch()





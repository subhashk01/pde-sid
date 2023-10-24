from sympy import symbols, diff, Function, sympify, lambdify, simplify
import re
import numpy as np
from util import find_highest_derivative


def differentiate_f(f_str, num_derivs, us):
    # f is a function of u and its derivatives
    # e.g. f = 'u*u_x'
    # calculates diff = d/dx^n f(u) for n = [0, num_derivs]
    # we then evaluate diff at each point in us (u are the P curves and their derivatives)
    # us is matrix of size P (num curvs) x num_derivs (of u) and N_p (num points evaluated at)

    # KNOWN BUG: DOES NOT WORK WHEN A COEFFICIENT IS 0.

    # Define the variable
    x = symbols('x')

    # Define the function u(x)
    u = Function('u')(x)

    # Compute derivatives up to some large number
    derivatives = {'u':u}
    highest_der = find_highest_derivative(f_str)

    derivs_data_num = us.shape[1]-1 # how many derivatives' data we actualy have
    assert(highest_der + num_derivs <= derivs_data_num) # need at least as much data as derivs

    deriv_list = [u]
    for i in range(1, 1+derivs_data_num):
        deriv_str = 'u_' + 'x' * i
        deriv_sym = diff(u, x, i) # ith derivative of u wrt x
        derivatives[deriv_str] = deriv_sym
        deriv_list.append(deriv_sym)

    # Convert string to SymPy expression
    past_f = None
    f_deriv = []
    for _ in range(num_derivs+1):
        if past_f is None: # first iteration. want to generate f as sympy expression
            f = sympify(f_str)
            for deriv_str, deriv_sym in derivatives.items():
                f = f.subs(deriv_str, deriv_sym)
        else:
            f = diff(past_f, x)
        f_func = lambdify([deriv_list], f) # makes useable function out of derivative
        f_vals = []
        for p in range(us.shape[0]): # might be a faster way to do this. evaluates derivativ
            func_val = f_func(us[p])
            if type(func_val) == int and func_val == 0: # only happens when 0 is the coefficient
                func_val = [0]*us.shape[2]
            f_vals.append(func_val) # gives all N_p values for each of the derivatives in u
        f_deriv.append(f_vals)
        past_f = f

    f_deriv = np.array(f_deriv)

    f_deriv = np.transpose(f_deriv, (1,0,2))
    assert(f_deriv.shape == (us.shape[0], num_derivs + 1, us.shape[2])) # should be P, num_derivs, N_p
    
    return f_deriv # returns (P, num_der, N_p)

def grad_basis_function(b, us):
    # b is a string that represents b(u'). ex. b(u') = u*u_xxxx+u_xxx
    # we want to calculate gradB w.r.t each u (u, u_x, u_xx, u_xxx, etc.)

    der = find_highest_derivative(b) # how many derivatives we need to take
    der_data = us.shape[1]-1 # how many derivatives we actualy have
    assert(der <= der_data) # need as much data as derivs required
    terms = ['u'] + ['u_'+'x'*d for d in range(1,der+1)]
    derivatives = [symbols(term) for term in terms]
    b_func = sympify(b)
    b_deriv = []
    for deriv in derivatives: # each term of b(u')
        deriv_func = diff(b_func,deriv)
        deriv_func = lambdify([derivatives], deriv_func)
        b_vals = []
        for p in range(us.shape[0]):
            val = deriv_func(us[p][:der+1]) # only takes as many derivatives as we require
            if type(val)!=list: # if its always a constant it will come out as an int
                val = val*np.ones((us.shape[2])) # N_p values
            b_vals.append(val) # P x N_p

        b_deriv.append(b_vals) # for each deriv
    b_deriv = np.array(b_deriv)
    b_deriv = np.transpose(b_deriv, (1,0,2))

    assert(b_deriv.shape == (us.shape[0], der + 1, us.shape[2])) # should be num_derivs, P, N_p
    return b_deriv  # size P, num_derivs, N_p


def calculate_matrix_columns_presum(b,f,us):
    b_der = find_highest_derivative(b)
    fs = differentiate_f(f, b_der, us)
    bs = grad_basis_function(b, us)
    assert(bs.shape == fs.shape) # should be P, num_derivs, N_p
    column_presum = np.einsum('ijk,ijk->ik', bs, fs) # should be P, N_p
    assert(column_presum.shape == (us.shape[0], us.shape[2])) # should be P, N_p
    return column_presum

def calculate_matrix_columns(b, f, us):
    # calculates each column of G (g(u') for each b(u') for all P points)

    column_presum = calculate_matrix_columns_presum(b,f,us)
    column = np.sum(column_presum, axis=1) # sum across N_p
    assert(column.shape == (us.shape[0],)) # should be P
    return column

def create_matrix(bs, f, us):
    # creates the matrix G (P x num_bs)
    columns = []
    for b in bs:
        column = calculate_matrix_columns(b, f, us)
        columns.append(column)
    columns = np.array(columns)
    columns = np.transpose(columns)
    assert(columns.shape == (us.shape[0], len(bs))) # should be P, num_bs
    return columns



if __name__ ==  '__main__':
    f = '-u_xxx+6*u*u_x'
   
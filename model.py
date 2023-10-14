import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import minimize
from calculate_G import create_matrix
import copy
from util import read_bases, check_function_integral

    
    
def find_cq(f, bases, check_trivial_bases = True, check_trivial_solutions = True, seed = 0):
    # assumes bases is of the form u
    np.random.seed(seed)

    # load data
    print("#### Loading data ####")

    us = np.load('test_curves.npy')

    

    bases = np.array(bases)
    if check_trivial_bases:
        print('#### Removing Trivial Bases ####')
        trivial = check_trivial(bases)
        if np.sum(trivial) != 0:
        
            print('Removed: ', bases[trivial==1])
        
            bases = bases[trivial==0]
            print('Remaining: ', bases)
        


    #### Computing bases and gradients #####
    print("#### Computing bases and gradients #####")
    f_grad_prod = create_matrix(bases, f, us)
    results = svd_and_sparsify(f_grad_prod)
    results['bases'] = bases
    if check_trivial_solutions:
        print('#### Calculating Number of Non Trivial Conserved Quantities ####')
        solutions = threshold_and_format(bases, results['sol_cq_sparse'])
        
        trivials = check_trivial(solutions)
        print(f'{np.sum(trivials==1)} Trivial Solutions')
        print(f'{np.sum(trivials==0)} Non Trivial Solutions')

        cq = 0
        for solution in np.array(solutions)[trivials==0]:
            cq += 1
            print(f'CQ{cq}: {solution}')
        results['trivials'] = trivials
    return results


def svd_and_sparsify(G, tol_cq=1e-6 ):
    results = {}

    #### Solving thetas ####
    print("#### Solving thetas ####")
    u, s, v = np.linalg.svd(G)

    u,s,v = u[::-1], s[::-1], v[::-1] #orders everything from smallest to largest SV, since that's what we care about

    results['s_cq_nonorm'] = copy.deepcopy(s)
    s = s/np.sum(s)
    print(s)
    ### out: s
    results['s_cq'] = copy.deepcopy(s)
    num_cq = np.sum(s<tol_cq) # tol_cq is a threshold (1e-8). sees how many significant cqs we have
    print('Number of Conserved Quantities: ', num_cq)
    #num_cq = len(v) # testing out sparsifying the whole matrix
    solutions = v[:num_cq]

    # np.einsum("ij,kj->ki", f_grad_prod, solutions) # check solutions
    ### out: solutions
    print("#### Sparsifying thetas ####")
    results['sol_cq'] = copy.deepcopy(solutions)
    sparse_sol = sparsify(solutions)
    results['sol_cq_sparse'] = copy.deepcopy(sparse_sol)


    non_solutions = v[num_cq:]
    results['non_sol_cq'] = copy.deepcopy(non_solutions)
    #### Sparsifying thetas ####



    sparse_non_sol = sparsify(non_solutions)
    results['non_sol_cq_sparse'] = copy.deepcopy(sparse_non_sol)
    
    
    return results


def sparsify(solutions, tol_dep=1e-4, seed=0, sparse_run=10, sparse_tol=1e-32, max_iter=100):
    n = len(solutions)
    def vector2orth(V):
        V = V.reshape(n, n)
        I = np.eye(n, n)
        A = (V-V.T)/2 # creates skew symmetric matrix
        Q = np.matmul(I-A, np.linalg.inv(I+A)) # cayley transform (makes skew symmetric go to orthogonal)
        return Q

    def L1(V):
        Q = vector2orth(V)
        sp = np.matmul(Q, solutions)
        l1 = np.sum(np.abs(sp))
        return l1

    grad_L1 = grad(L1)


    sols = []
    sol_funcs = []
    # try different seeds
    num_seed = sparse_run # how many times we'll do sparisy turn
    for i in range(num_seed):
        print("{}/{}".format(i,num_seed))
        seed = i
        np.random.seed(seed)
        V = np.random.randn(n, n)
        sol = minimize(L1, V, method = "L-BFGS-B", jac=grad_L1, tol=1e-10, options={'maxiter':max_iter})
        sol_funcs.append(sol.fun)
        sols.append(sol.x)

    sol_funcs = np.array(sol_funcs)
    winner = np.argmin(sol_funcs)
    sol_func = sols[winner]
    solutions = np.matmul(vector2orth(sol_func), solutions)
    ### out: solutions
    if len(solutions.shape) == 1:
        solutions = np.array([solutions])
    return solutions


def check_trivial(bases):
    # returns an array x of size bases that has a 0 if the base is non trivial and 1 if its trivia;
    x = np.zeros(len(bases))

    us = np.load('test_curves.npy')
    values = []
    for b in bases:
        val = check_function_integral(b, us)
        values.append(val)
    values = np.transpose(np.array(values))
    trivial_svd = svd_and_sparsify(values, 1e-6, 10)
    if trivial_svd is None: # no trivials
        return x
    
    trivial_sp = trivial_svd['sol_cq_sparse']

    for j in range(trivial_sp.shape[1]):
        col = trivial_sp[:, j]
        if np.sum((col > 0.99) & (np.abs(col - 1) < 1e-3)) == 1:
            x[j] = 1
    x = x.astype(int) 
    return x




def threshold_and_format(b, a, threshold=1e-1):
    assert(a.shape[1] == len(b))
    # Threshold the coefficients
    a[np.abs(a) < threshold] = 0
    
    # Initialize list to hold conserved quantities as strings
    conserved_quantities = []

    
    # Loop through each row in the thresholded array `a`
    for row in a:
        terms = []
        for coef, basis in zip(row, b):
            # Only include terms with non-zero coefficients
            if coef != 0:
                # check if coef is within threshold of a whole number
                if np.abs(coef - np.round(coef)) < threshold:
                    term = f"{coef:.0f}*({basis})"
                else:
                    term = f"{coef:.2f}*({basis})"
                terms.append(term)
        
        # Join the terms into a single string and append to list
        conserved_quantity = " + ".join(terms)
        
        # Skip empty strings
        if conserved_quantity:
            conserved_quantities.append(conserved_quantity)
    
    return conserved_quantities
    

if __name__ == '__main__':
    f = 'u_xxx-6*u*u_x'
    b = read_bases()
    find_cq(f,b)

    

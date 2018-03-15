import torch
from torch.autograd import Variable
from gpytorch.lazy import LazyVariable, NonLazyVariable

def pivoted_cholesky(matrix, max_iter, error_tol=1e-5):
    matrix_size = matrix.size(-1)
    matrix_diag = matrix.diag()

    # TODO: This check won't be necessary in PyTorch 0.4
    if isinstance(matrix_diag, torch.autograd.Variable):
        matrix_diag = matrix_diag.data

    error = torch.norm(matrix_diag, 1)
    permutation = matrix_diag.new(matrix_size).long()
    torch.arange(0, matrix_size, out=permutation)

    m = 0
    # TODO: pivoted_cholesky should take tensor_cls and use that here instead
    L = matrix_diag.new(max_iter, matrix_size).zero_()
    while m < max_iter and error > error_tol:
        max_diag_value, max_diag_index = torch.max(matrix_diag[permutation][m:], 0)
        max_diag_index = max_diag_index + m

        pi_m = permutation[m]
        permutation[m] = permutation[max_diag_index][0]
        permutation[max_diag_index] = pi_m

        pi_m = permutation[m]

        L_m = L[m] # Will be all zeros -- should we use torch.zeros?
        L_m[pi_m] = torch.sqrt(max_diag_value)[0]

        row = matrix[pi_m]

        if isinstance(row, torch.autograd.Variable):
            row = row.data

        pi_i = permutation[m + 1:]
        L_m[pi_i] = row[pi_i]
        if m > 0:
            L_prev = L[:m].index_select(1, pi_i)
            L_m[pi_i] -= torch.sum(L[:m, pi_m].unsqueeze(1) * L_prev, dim=0)
        L_m[pi_i] /= L_m[pi_m]

        matrix_diag[pi_i] = matrix_diag[pi_i] - (L_m[pi_i] ** 2)
        L[m] = L_m

        error = torch.sum(matrix_diag[permutation[m + 1:]])
        m = m + 1

    return L[:m, :]

def pivoted_cholesky_batch(matrix, max_iter, error_tol=1e-5):
    # matrix is now batch_size x n x n
    batch_size = matrix.size(0)
    matrix_size = matrix.size(-1)

    # Need to get diagonals. This is easy if it's a LazyVariable, since
    # LazyVariable.diag() operates in batch mode.
    if isinstance(matrix, LazyVariable):
        matrix_diags = matrix.diag
    elif isinstance(matrix, Variable):
        matrix_diags = NonLazyVariable(matrix).diag()
    elif torch.is_tensor(matrix):
        matrix_diags = NonLazyVariable(Variable(matrix)).diag()

    # TODO: This check won't be necessary in PyTorch 0.4
    if isinstance(matrix_diag, torch.autograd.Variable):
        matrix_diag = matrix_diag.data

    # matrix_diag is now batch_size x n

    errors = torch.norm(matrix_diag, 1, dim=1)
    permutation = matrix_diag.new(matrix_size).long()
    torch.arange(0, matrix_size, out=permutation)
    permutation = permutation.repeat(batch_size, 1)

    m = 0
    # TODO: pivoted_cholesky should take tensor_cls and use that here instead
    L = matrix_diag.new(batch_size, max_iter, matrix_size).zero_()

    full_batch_slice = torch.arange(batch_size).long()
    while m < max_iter and error > error_tol:
        permuted_diags = torch.gather(matrix_diag, 1, permutation)[:, m:]
        max_diag_values, max_diag_indices = torch.max(permuted_diags, 1)

        max_diag_indices = max_diag_indices + m

        # Swap pi_m and pi_i in each row, where pi_i is the element of the permutation
        # corresponding to the max diagonal element
        old_pi_m = permutation[:, m]
        new_pi_m = permutation[full_batch_slice, max_diag_indices]
        permutation[:, m] = new_pi_m
        permutation[full_batch_slice, max_diag_indices] = old_pi_m

        pi_m = permutation[:, m]

        L_m = L[:, m] # Will be all zeros -- should we use torch.zeros?
        L_m[full_batch_slice, pi_m] = torch.sqrt(max_diag_value)[0]

        row = matrix[full_batch_slice, pi_m, :]

        if isinstance(row, torch.autograd.Variable):
            row = row.data

        #*** TODO: UP TO HERE CONVERTING TO BATCH MODE ***
        pi_i = permutation[m + 1:]
        L_m[pi_i] = row[pi_i]
        if m > 0:
            L_prev = L[:m].index_select(1, pi_i)
            L_m[pi_i] -= torch.sum(L[:m, pi_m].unsqueeze(1) * L_prev, dim=0)
        L_m[pi_i] /= L_m[pi_m]

        matrix_diag[pi_i] = matrix_diag[pi_i] - (L_m[pi_i] ** 2)
        L[m] = L_m

        error = torch.sum(matrix_diag[permutation[m + 1:]])
        m = m + 1

    return L[:m, :]

def woodbury_factor(low_rank_mat, shift):
    """
    Given a low rank (k x n) matrix V and a shift, returns the
    matrix R so that
        R = (I_k + 1/shift VV')^{-1}V
    to be used in solves with (V'V + shift I) via the Woodbury formula
    """
    n = low_rank_mat.size(-1)
    k = low_rank_mat.size(-2)
    shifted_mat = (1 / shift) * low_rank_mat.matmul(low_rank_mat.t())
    shifted_mat = shifted_mat + shifted_mat.new(k).fill_(1).diag()

    R = torch.potrs(low_rank_mat, shifted_mat.potrf())

    return R


def woodbury_solve(vector, low_rank_mat, woodbury_factor, shift):
    """
    Solves the system of equations:
        (sigma*I + VV')x = b
    Using the Woodbury formula.

    Input:
        - vector (size n) - right hand side vector b to solve with.
        - woodbury_factor (k x n) - The result of calling woodbury_factor on V
          and the shift, \sigma
        - shift (scalar) - shift value sigma
    """
    right = (1 / shift) * low_rank_mat.t().matmul(woodbury_factor.matmul(vector))
    return (1 / shift) * (vector - right)

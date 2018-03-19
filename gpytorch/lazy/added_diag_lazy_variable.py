import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from .diag_lazy_variable import DiagLazyVariable
from .non_lazy_variable import NonLazyVariable
from .psd_sum_lazy_variable import PsdSumLazyVariable


class AddedDiagLazyVariable(PsdSumLazyVariable):
    def __init__(self, lazy_var, diag_lazy_var):
        if not isinstance(lazy_var, LazyVariable):
            lazy_var = NonLazyVariable(lazy_var)
        if not isinstance(diag_lazy_var, DiagLazyVariable):
            diag_lazy_var = DiagLazyVariable(diag_lazy_var)
        if lazy_var.size() != diag_lazy_var.size():
            raise RuntimeError('Size of lazy variable (%s) doesn\'t match size of '
                               'diag (%s).' % (lazy_var.size(), diag_lazy_var.size()))

        super(AddedDiagLazyVariable, self).__init__(lazy_var, diag_lazy_var)

    @property
    def _lazy_var_root(self):
        if not hasattr(self, '_lazy_var_root_memo'):
            self._lazy_var_root_memo = self.lazy_vars[0].root_decomposition()
        return self._lazy_var_root_memo

    @property
    def _diag(self):
        return self.lazy_vars[1]._diag

    def inv_matmul(self, rhs):
        # Use the Woodbury formula
        # (RR^T + D)^{−1} = D^{−1} − D^{−1} R (I + R^T D^{−1} R)^{−1} R^T D^{−1}
        # Where RR^T is the root decomp of self._lazy_var, and D is self._diag
        root = self._lazy_var_root
        inv_diag = torch.reciprocal(self._diag)
        inv_diag_root = root.mul(inv_diag.unsqueeze(-1))

        if not hasattr(self, '_woodbury_cache'):
            eye_diag = Variable(root.data.new([1]))
            if root.ndimension() == 3:
                eye_diag.data.unsqueeze_(0)
            eye_diag.data = eye_diag.data.expand_as(self._diag)

            inner_mat = PsdSumLazyVariable(
                NonLazyVariable(root.transpose(-1, -2).matmul(inv_diag_root)),
                DiagLazyVariable(eye_diag),
            )
            self._woodbury_cache = inner_mat.inv_matmul(
                inv_diag_root.transpose(-1, -2)
            )

        left_term = inv_diag.unsqueeze(-1).mul(rhs)
        right_term = inv_diag_root.matmul(
            self._woodbury_cache.matmul(rhs)
        )
        return left_term - right_term

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False):
        # Use the Matrix determinant lemma
        # |(RR^T + D)| = |I + R^T D^{-1} R| |D|
        # Where RR^T is the root decomp of self._lazy_var, and D is self._diag
        root = self._lazy_var_root
        inv_diag = torch.reciprocal(self._diag)
        inv_diag_root = root.mul(inv_diag.unsqueeze(-1))

        # Inner matrix: I + R^T D^{-1} R
        eye_diag = Variable(root.data.new([1]))
        if root.ndimension() == 3:
            eye_diag.data.unsqueeze_(0)
        eye_diag.data = eye_diag.data.expand_as(self._diag)
        inner_mat = PsdSumLazyVariable(
            NonLazyVariable(root.transpose(-1, -2).matmul(inv_diag_root)),
            DiagLazyVariable(eye_diag),
        )

        # Get the inner quad form and the 
        inner_inv_quad_rhs = None
        if inv_quad_rhs is not None:
            inner_inv_quad_rhs = inv_diag_root.transpose(-1, -2).matmul(inv_quad_rhs)
        inner_inv_quad_term, inner_log_det_term = inner_mat.inv_quad_log_det(
            inner_inv_quad_rhs,
            log_det
        )

        # Get the final terms
        inv_quad_term = None
        log_det_term = None
        if inv_quad_rhs is not None:
            left_term = (
                inv_quad_rhs.
                mul(self._diag.unsqueeze(-1)).
                mul(inv_quad_rhs).
                sum(-1).sum(-1, keepdim=(root.ndimension() == 3))
            )
            right_term = inner_inv_quad_term
            inv_quad_term = left_term - right_term
        if log_det:
            log_det_term = inner_log_det_term + self._diag.log().sum(-1)

        return inv_quad_term, log_det_term

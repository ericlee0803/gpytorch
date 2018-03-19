import os
import math
import numpy as np
import gpytorch
import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import AddedDiagLazyVariable, CholLazyVariable, DiagLazyVariable
from gpytorch.utils import approx_equal


class TestAddedDiagLazyVariable(unittest.TestCase):
    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        chol = torch.Tensor([
            [3, 0, 0, 0, 0],
            [-1, 2, 0, 0, 0],
            [1, 4, 1, 0, 0],
            [0, 2, 3, 2, 0],
            [-4, -2, 1, 3, 4],
        ])
        diag = torch.Tensor([3, 2, 1, 1, 2])
        vecs = torch.randn(5, 2)

        self.chol_var = Variable(chol, requires_grad=True)
        self.chol_var_copy = Variable(chol, requires_grad=True)
        self.diag_var = Variable(diag, requires_grad=True)
        self.diag_var_copy = Variable(diag, requires_grad=True)
        self.lazy_var = AddedDiagLazyVariable(
            CholLazyVariable(self.chol_var),
            DiagLazyVariable(self.diag_var),
        )
        self.actual_mat = sum([
            self.chol_var_copy.matmul(self.chol_var_copy.transpose(-1, -2)),
            self.diag_var_copy.diag(),
        ])
        self.vecs = Variable(vecs, requires_grad=True)
        self.vecs_copy = Variable(vecs, requires_grad=True)

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_matmul(self):
        # Forward
        res = self.lazy_var.matmul(self.vecs)
        actual = self.actual_mat.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(*self.vecs.size())
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.chol_var.grad.data, self.chol_var_copy.grad.data))
        self.assertTrue(approx_equal(self.diag_var.grad.data, self.diag_var_copy.grad.data))
        self.assertTrue(approx_equal(self.vecs.grad.data, self.vecs_copy.grad.data))

    def test_inv_matmul(self):
        # Forward
        res = self.lazy_var.inv_matmul(self.vecs)
        actual = self.actual_mat.inverse().matmul(self.vecs_copy)
        self.assertLess(torch.max((res.data - actual.data).abs() / actual.data.norm()), 1e-2)

    def test_inv_quad_log_det(self):
        # Forward
        with gpytorch.settings.num_trace_samples(1000):
            res_inv_quad, res_log_det = self.lazy_var.inv_quad_log_det(inv_quad_rhs=self.vecs,
                                                                       log_det=True)
        res = res_inv_quad + res_log_det
        actual_inv_quad = self.actual_mat.inverse().matmul(self.vecs_copy).mul(self.vecs_copy).sum()
        actual = actual_inv_quad + math.log(np.linalg.det(self.actual_mat.data))
        self.assertLess(((res.data - actual.data) / actual.data).abs()[0], 1e-1)

    def test_diag(self):
        res = self.lazy_var.diag()
        actual = self.actual_mat.diag()
        self.assertTrue(approx_equal(res.data, actual.data))

    def test_getitem(self):
        res = self.lazy_var[2:4, -2]
        actual = self.actual_mat[2:4, -2]
        self.assertTrue(approx_equal(res.data, actual.data))

    def test_evaluate(self):
        res = self.lazy_var.evaluate()
        actual = self.actual_mat
        self.assertTrue(approx_equal(res.data, actual.data))


class TestAddedDiagLazyVariableBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        chol = torch.Tensor([
            [
                [3, 0, 0, 0, 0],
                [-1, 2, 0, 0, 0],
                [1, 4, 1, 0, 0],
                [0, 2, 3, 2, 0],
                [-4, -2, 1, 3, 4],
            ], [
                [2, 0, 0, 0, 0],
                [3, 1, 0, 0, 0],
                [-2, 3, 2, 0, 0],
                [-2, 1, -1, 3, 0],
                [-4, -4, 5, 2, 3],
            ],
        ])
        diag = torch.Tensor([
            [3, 2, 1, 1, 2],
            [2, 1, 2, 2, 3],
        ])
        vecs = torch.randn(2, 5, 3)

        self.chol_var = Variable(chol, requires_grad=True)
        self.chol_var_copy = Variable(chol, requires_grad=True)
        self.diag_var = Variable(diag, requires_grad=True)
        self.diag_var_copy = Variable(diag, requires_grad=True)
        self.lazy_var = AddedDiagLazyVariable(
            CholLazyVariable(self.chol_var),
            DiagLazyVariable(self.diag_var),
        )
        self.actual_mat = sum([
            self.chol_var_copy.matmul(self.chol_var_copy.transpose(-1, -2)),
            torch.cat([
                self.diag_var_copy[0].diag().unsqueeze(0),
                self.diag_var_copy[1].diag().unsqueeze(0),
            ], 0),
        ])
        self.actual_mat_inv = torch.cat([
            self.actual_mat[0].inverse().unsqueeze(0),
            self.actual_mat[1].inverse().unsqueeze(0),
        ], 0)

        self.vecs = Variable(vecs, requires_grad=True)
        self.vecs_copy = Variable(vecs, requires_grad=True)

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_matmul(self):
        # Forward
        res = self.lazy_var.matmul(self.vecs)
        actual = self.actual_mat.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(*self.vecs.size())
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.chol_var.grad.data, self.chol_var_copy.grad.data))
        self.assertTrue(approx_equal(self.diag_var.grad.data, self.diag_var_copy.grad.data))
        self.assertTrue(approx_equal(self.vecs.grad.data, self.vecs_copy.grad.data))

    def test_inv_matmul(self):
        # Forward
        res = self.lazy_var.inv_matmul(self.vecs)
        actual = self.actual_mat_inv.matmul(self.vecs_copy)
        self.assertLess(torch.max((res.data - actual.data).abs() / actual.data.norm()), 1e-2)

    def test_inv_quad_log_det(self):
        # Forward
        with gpytorch.settings.num_trace_samples(1000):
            res_inv_quad, res_log_det = self.lazy_var.inv_quad_log_det(inv_quad_rhs=self.vecs, log_det=True)
        res = res_inv_quad + res_log_det
        actual_inv_quad = (
            self.actual_mat_inv.
            matmul(self.vecs_copy).
            mul(self.vecs_copy).
            sum(-1).sum(-1)
        )
        actual_log_det = Variable(torch.Tensor([
            math.log(np.linalg.det(self.actual_mat[0].data)),
            math.log(np.linalg.det(self.actual_mat[1].data)),
        ]))
        actual = actual_inv_quad + actual_log_det
        self.assertLess(torch.max((res.data - actual.data).abs() / actual.data.norm()), 1e-1)

    def test_diag(self):
        res = self.lazy_var.diag()
        actual = torch.cat([
            self.actual_mat[0].diag().unsqueeze(0),
            self.actual_mat[1].diag().unsqueeze(0),
        ], 0)
        self.assertTrue(approx_equal(res.data, actual.data))

    def test_getitem(self):
        res = self.lazy_var[1, 2:4, -2]
        actual = self.actual_mat[1, 2:4, -2]
        self.assertTrue(approx_equal(res.data, actual.data))

    def test_evaluate(self):
        res = self.lazy_var.evaluate()
        actual = self.actual_mat
        self.assertTrue(approx_equal(res.data, actual.data))


if __name__ == '__main__':
    unittest.main()

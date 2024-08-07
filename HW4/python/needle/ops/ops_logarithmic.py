from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=self.axes, keepdims=True)
        max_Z_reduced = Z.max(axis=self.axes)
        return array_api.log(array_api.summation(array_api.exp(Z - max_Z).broadcast_to(Z.shape), axis=self.axes)) + max_Z_reduced
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        z_max_dim = Tensor(z.realize_cached_data().max(self.axes, keepdims=True), device=z.device)
        z_exp = exp(z + (-z_max_dim).broadcast_to(z.shape))
        z_exp_sum = summation(z_exp, axes=self.axes)
        grad_z_exp_sum = out_grad / z_exp_sum
        ori_shape = z.shape
        sum_shape = range(len(z.shape)) if self.axes is None else self.axes
        now_shape = list(ori_shape)
        for i in sum_shape:
            now_shape[i] = 1
        return reshape(grad_z_exp_sum, now_shape).broadcast_to(ori_shape) * z_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


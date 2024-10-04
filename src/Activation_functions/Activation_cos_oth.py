import torch
import torch.nn as nn
from torch import cos, sin
# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias):
        
        ctx.save_for_backward(input, weight, bias)
        
        return cos(input).mm(weight) + bias

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        
        input, weight, bias = ctx.saved_tensors

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        grad_input = grad_output * (-sin(input)*weight) if ctx.needs_input_grad[0] else None
        grad_weight = grad_output * cos(input) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output if bias is not None and ctx.needs_input_grad[2] else None

        return grad_input, grad_weight, grad_bias
    
class Cos(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

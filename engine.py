from typing import List
from abc import ABC, abstractstaticmethod

from collections import defaultdict, namedtuple
import numpy as np

Variable = namedtuple("Variable", ["name", "value"])


def _reset_graph():
    global GRAPH
    GRAPH = defaultdict(dict)


_reset_graph()


class Function(ABC):
    @abstractstaticmethod
    def forward(input: List[Variable]) -> Variable:
        pass

    @abstractstaticmethod
    def backward(grad_output: np.ndarray) -> List[np.ndarray]:
        pass


def _make_variable(input_vars, output):
    varname = str(hash("".join(GRAPH.keys())))
    output_var = Variable(varname, output)

    for x in input_vars:
        if "children" not in GRAPH[x.name]:
            GRAPH[x.name]["children"] = [output_var]
        else:
            GRAPH[x.name]["children"].append(output_var)

    GRAPH[varname]["parents"] = [x for x in input_vars]
    return output_var


class Add(Function):
    @staticmethod
    def forward(input):
        output = _make_variable(
            input_vars=input,
            output=sum(x.value for x in input),
        )
        GRAPH[output.name]["ctx"] = Add, len(input)
        return output

    @staticmethod
    def backward(grad_output, ctx):
        return [grad_output] * ctx


class Matmul(Function):
    @staticmethod
    def forward(input):
        A, B = input
        output = _make_variable(input, A.value @ B.value)
        GRAPH[output.name]["ctx"] = Matmul, [A.value, B.value]
        return output

    @staticmethod
    def backward(grad_output, ctx):
        A, B = ctx        
        grads = [grad_output @ B.T, A.T @ grad_output]
        return grads


class Softmax(Function):
    @staticmethod
    def forward(input_vars):
        (input,) = input_vars
        x = input.value
        assert x.ndim == 1  # only handle 1d case
        x = x - x.max()
        acts = np.exp(x)
        probs = acts / np.sum(acts)

        output = _make_variable(input_vars, probs)
        GRAPH[output.name]["ctx"] = Softmax, probs
        return output

    @staticmethod
    def backward(grad_output, ctx):
        probs = ctx
        jacrev = np.diag(probs) - probs[None] * probs[:, None]
        return [jacrev @ grad_output]


class Reshape(Function):
    @staticmethod
    def forward(input_vars, shape=(-1,)):
        (input,) = input_vars
        x = input.value
        orig_shape = x.shape

        output = x.reshape(shape)
        output_var = _make_variable(input_vars, output)
        GRAPH[output_var.name]["ctx"] = Reshape, orig_shape
        return output_var

    @staticmethod
    def backward(grad_output, ctx):
        return [grad_output.reshape(ctx)]


def elementwise_factory(f, f_prime):
    class Elementwise(Function):
        @staticmethod
        def forward(input):
            (x,) = input
            output = _make_variable(input, f(x.value))
            GRAPH[output.name]["ctx"] = Elementwise, f_prime(x.value)
            return output

        @staticmethod
        def backward(grad_output, ctx):
            return [ctx * grad_output]
    return Elementwise


ReLU = elementwise_factory(
    lambda x: x * (x > 0).astype(np.float32),
    lambda x: (x > 0).astype(np.float32),
)
Exp = elementwise_factory(
    lambda x: np.e**x,
    lambda x: np.e**x,
)
Log = elementwise_factory(
    lambda x: np.log(x),
    lambda x: 1 / x,
)


def backward(variable):
    assert "children" not in GRAPH[variable.name]
    work = [variable]
    
    # seed
    GRAPH[variable.name]["grad"] = np.ones_like(variable.value)

    while work:
        element = work.pop()


        element_data = GRAPH[element.name]
        assert 'children' not in element_data or not element_data['children']

        if 'parents' not in element_data or not element_data['parents']:
            pass
        else:
            backward_fn, ctx = element_data["ctx"]
            grad_outputs = backward_fn.backward(element_data["grad"], ctx)

            assert len(grad_outputs) == len(element_data["parents"])

            for grad_output, parent in zip(grad_outputs, element_data["parents"]):
                if "grad" not in GRAPH[parent.name]:
                    GRAPH[parent.name]["grad"] = grad_output
                else:
                    assert GRAPH[parent.name]["grad"].shape == grad_output.shape 
                    GRAPH[parent.name]["grad"] += grad_output

                GRAPH[parent.name]["children"] = [
                    c
                    for c in GRAPH[parent.name]["children"]
                    if c.name != element.name
                ]

                if not GRAPH[parent.name]["children"]:
                    work.append(parent)

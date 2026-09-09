"""
Week 6: Calculus — Automatic Differentiation Engine & Backpropagation (Micrograd-Style)

Implements a reverse-mode automatic differentiation engine from first principles.
Builds the computational DAG and propagates gradients using the chain rule:
dL/dx = (dL/dy) * (dy/dx)
"""

import math
from typing import Set, List, Union


class Scalar:
    """
    A computational graph node wrapping a scalar value and its accumulated gradient.
    """

    def __init__(self, data: float, _children: tuple = (), _op: str = ''):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward = lambda: None
        self._prev: Set['Scalar'] = set(_children)
        self._op: str = _op

    def __repr__(self) -> str:
        return f"Scalar(data={self.data:.4f}, grad={self.grad:.4f})"

    # -------------------------------------------------------------
    # Forward Arithmetic & Local Chain Rule Closures
    # -------------------------------------------------------------
    def __add__(self, other: Union['Scalar', float]) -> 'Scalar':
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            # d(x + y)/dx = 1, d(x + y)/dy = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Union['Scalar', float]) -> 'Scalar':
        return self + other

    def __mul__(self, other: Union['Scalar', float]) -> 'Scalar':
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            # d(x * y)/dx = y, d(x * y)/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Scalar', float]) -> 'Scalar':
        return self * other

    def __neg__(self) -> 'Scalar':
        return self * -1.0

    def __sub__(self, other: Union['Scalar', float]) -> 'Scalar':
        return self + (-other)

    def __rsub__(self, other: Union['Scalar', float]) -> 'Scalar':
        return Scalar(other) - self

    def __truediv__(self, other: Union['Scalar', float]) -> 'Scalar':
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union['Scalar', float]) -> 'Scalar':
        return Scalar(other) / self

    def __pow__(self, power: Union[int, float]) -> 'Scalar':
        assert isinstance(power, (int, float)), "Only supporting int/float powers"
        out = Scalar(self.data ** power, (self,), f'**{power}')

        def _backward():
            # d(x^p)/dx = p * x^(p-1)
            self.grad += (power * (self.data ** (power - 1.0))) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Scalar':
        x = self.data
        out = Scalar(math.exp(x), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self) -> 'Scalar':
        out = Scalar(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Scalar':
        # 1 / (1 + exp(-x))
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Scalar(s, (self,), 'sigmoid')

        def _backward():
            # d(sigma)/dx = sigma * (1 - sigma)
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    # -------------------------------------------------------------
    # Reverse-Mode Backpropagation via Topological Sort
    # -------------------------------------------------------------
    def backward(self):
        """Executes full backpropagation through the computational graph."""
        topo: List[Scalar] = []
        visited: Set[Scalar] = set()

        def build_topo(v: Scalar):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed the base gradient: dL/dL = 1.0
        self.grad = 1.0
        # Traverse in reverse topological order (outputs to inputs)
        for node in reversed(topo):
            node._backward()


# =====================================================================
# Demonstration: Mini Neuron with Chain Rule
# =====================================================================
def neuron_demo():
    """
    Demonstrates forward and backward pass through an artificial neuron:
    f(x1, x2) = sigmoid(w1*x1 + w2*x2 + b)
    """
    x1 = Scalar(2.0)
    x2 = Scalar(0.0)
    w1 = Scalar(-3.0)
    w2 = Scalar(1.0)
    b = Scalar(6.88137)

    # Forward graph
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1_x2w2 = x1w1 + x2w2
    n = x1w1_x2w2 + b
    out = n.sigmoid()

    # Backpropagation
    out.backward()

    return {
        "output": out.data,
        "grad_x1": x1.grad,
        "grad_w1": w1.grad,
        "grad_b": b.grad
    }


if __name__ == "__main__":
    print("=== Autograd Computational Graph Demo ===")
    res = neuron_demo()
    print(f"Neuron Output: {res['output']:.4f}")
    print(f"Weight w1 grad: {res['grad_w1']:.4f}")
    print(f"Bias b grad:    {res['grad_b']:.4f}")

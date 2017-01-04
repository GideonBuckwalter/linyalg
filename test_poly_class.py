import pytest
from .poly import *
from .matrix import *

def test_characteristic_polynomial():
    M = Matrix([[1,0,0],
                [0,2,0],
                [0,0,3]])

    L = Polynomial.variable("L")
    p = (L-1)*(L-2)*(L-3)

    assert characteristic_polynomial(M) == p

def test_eigenvalues():
    M = Matrix([[1,0,0],
                [0,2,0],
                [0,0,3]])
    assert eigenvalues(M) == {1,2,3}

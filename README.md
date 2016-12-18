# linyalg
A student-made, human-readable, lightweight linear algebra library written in Python and having NumPy-like syntax.

<h3>Example: Using LinyAlg to perform a polynomial regression</h3>

```
>>> from linyalg.matrix import *
>>> Points = Matrix([[0, 0.1], [1, 1.2], [2, 3.7], [3, 9.5], [4, 17], [5, 25.6]])
>>> x, y = Points[:, 0], Points[:, 1]
>>> y
Matrix([[ 0.1],
        [ 1.2],
        [ 3.7],
        [ 9.5],
        [  17],
        [25.6]])
>>> degree = 2
>>> A = Matrix([[x_ ** e for e in range(degree+1)] for x_ in x])
>>> print(A)
[1, 0,  0]
[1, 1,  1]
[1, 2,  4]
[1, 3,  9]
[1, 4, 16]
[1, 5, 25]
6x3
>>> # Approximate a solution with 'A^T A Beta = A^T y'.
>>> Beta = inv(A.T @ A) @ A.T @ y
>>> Beta
Matrix([[ 0.03214285714286369],
        [0.028928571428559202],
        [  1.0267857142857135]])
>>> from linyalg.poly import *
>>> p = Polynomial(list(Beta))
>>> print(p)
0.03214285714286369 + 0.028928571428559202x + 1.0267857142857135x^2
```

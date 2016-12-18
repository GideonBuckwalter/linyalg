import operator
from functools import reduce
if __name__ != "__main__":
    from .poly import *
import copy

__author__ = "Gideon Buckwalter"

"""
Author: Gideon Buckwalter
Email: gideon.buckwalter@gmail.com
GitHub: @GideonBuckwalter
Creation Date: 29 June 2016
Version: Python 3.5.2
"""

class Matrix(object):
    def __init__(self, mat):
        """
        Pass a two-dimensional list containing the matrix data.
        """
        self.mat = mat
        self.m = len(mat)
        try:
            self.n = len(mat[0])
        except IndexError:
            raise IndexError("Matrix.__init__ was given a 2D list '{}' ".format(mat) +
                            "that did not have distinct rows.")
        self.size = (self.m, self.n)

    @property
    def T(self):
        """
        Returns the transpose of the matrix. Turns all rows into columns and all columns
        into rows.
        Syntax: B = A.T
        """
        return Matrix([list(self[:, i]) for i in range(self.n)])

    def __getitem__(self, key):
        """
        Defines bracket indexing of matrices.
        Syntax:
        row_2 = M[1]
        row_3 = M[2,:]
        ele_23 = M[1,2]
        """

        if type(key) is tuple:

            i, j = key
            if type(i) is slice:
                if type(j) is int:
                    j = slice(j, j+1, None)
                elif type(j) is not slice:
                    msg = "Invalid matrix index argument type: {}, {}"
                    raise TypeError(msg.format(type(i), type(j)))

                return Matrix([row[j] for row in self.mat[i]])

            elif type(i) is int:
                if type(j) is int:
                    return self.mat[i][j]
                elif type(j) is slice:
                    return Matrix([self.mat[i][j]])
                else:
                    msg = "Invalid matrix index argument type: {}, {}"
                    raise TypeError(msg.format(type(i), type(j)))
        
        elif type(key) in (int, slice):
            try:
                if type(key) is int:
                    return Matrix([self.mat[key]])
                else:
                    return Matrix(self.mat[key])
            except IndexError:
                msg = "Invalid slice/index '{}' given for matrix:\n{}."
                raise IndexError(msg.format(key, self))
        else:
            raise TypeError("Invalid matrix index type: {}".format(type(inp)))

    def __setitem__(self, key, value):
        """
        Allows mutable bracket assignments.
        Syntax: M[1] = Matrix([[3, 4, 5]])
                M[1, 2] = 100
        """
        if type(key) is tuple:
            i, j = key

            if type(i) is int:
                if type(j) is int:
                    self.mat[i][j] = value
                elif type(j) is slice:

                    # Used for error detection
                    j = format_slice(j, self, kind="col")

                    if value.size != (1, j.stop - j.start):
                        msg = "Slice bounds ({}, {}) did not match overwriting Matrix size {}."
                        raise IndexError(msg.format(i, j, value.size))

                    # value is something like Matrix([[5,3,2,6,1]])
                    self.mat[i][j] = value.mat[0]
                else:
                    msg = "Invalid matrix index argument type: {}, {}"
                    raise TypeError(msg.format(type(i), type(j)))
            elif type(i) is slice:
                if type(j) is int:
                    # value is something like Matrix([[1],[2],[3]])
                    M = copy.deepcopy(self).T
                    M[j,i] = value.T # Reuse above code on self's transpose
                    self.mat = M.T.mat
                elif type(j) is slice:

                    # Used for error detection
                    i = format_slice(i, self, kind="row")
                    j = format_slice(j, self, kind="col")
                    if value.size != (i.stop - i.start, j.stop - j.start):
                        msg = "Slice bounds ({}, {}) did not match overwriting Matrix size {}."
                        raise IndexError(msg.format(i, j, value.size))

                    # value is something like Matrix([[5,3,2],[1,8,2],[0,4,3]])
                    for i_star in range(i.start, i.stop):
                        self[i_star, j] = value[i_star-i.start]
                else:
                    msg = "Invalid matrix index argument type: {}, {}"
                    raise TypeError(msg.format(type(i), type(j)))
        elif type(key) in (int, slice):
            self.mat[key] = value.mat
        else:
            raise TypeError("Invalid matrix index type: {}".format(type(inp)))

    def list_rows(self):
        """
        Returns an iterable of all the row LISTS in the matrix.
        >>> M = Matrix([[1,2,3],[4,5,6]])

        >>> list(M.list_rows())
        [[1,2,3], [4,5,6]]
        """
        return iter(self.mat)

    def list_cols(self):
        """
        Returns and iterable of all the column LISTS in the matrix.
        >>> M = Matrix([[1,2,3],[4,5,6]])

        >>> list(M.list_cols())
        [[1,4],[2,5],[3,6]]
        """
        return self.T.list_rows()

    def row_vecs(self):
        """
        Returns an iterable of the rows represented as matrices.
        >>> M = Matrix([[1,2,3],[4,5,6],[7,8,9]])

        >>> list(M.row_vecs())
        [Matrix([[1,2,3]]), Matrix([[4,5,6]]), Matrix([[7,8,9]])]
        """
        return (Matrix([row]) for row in self.list_rows())

    def col_vecs(self):
        """
        Returns an iterable of the column vectors represented as matrices.
        >>> M = Matrix([[1,2,3],[4,5,6],[7,8,9]])

        >>> list(M.col_vecs())
        [Matrix([[1],[4],[7]]), Matrix([[2],[5],[8]]), Matrix([[3],[6],[9]])]
        """
        return (row_vec.T for row_vec in self.T.row_vecs())

    def under(self, func):
        """
        A.under(math.cos) will return a copy of Matrix A where each element
        has been replaced with the cosine of that element. Matrix.under only
        works for single-variable functions.
        Syntax: A.under(math.cos)
        """
        return Matrix([[func(x) for x in row] for row in self.mat])

    def __str__(self):
        """
        Returns a string representation of the matrix.
        Syntax: str(A)
                print(A)
        """
        str_mat = self.under(str)

        col_spacings = [max(map(len, col)) for col in str_mat.list_cols()]

        new_mat = [[ele.rjust(padding) for ele, padding in zip(row, col_spacings)]
                                       for row in str_mat.list_rows()]

        rep = ['[' + ', '.join(row) + ']' for row in new_mat] + \
              [str(self.m) + "x" + str(self.n)]
        return "\n".join(rep)

    def __repr__(self):
        repr_mat = self.under(repr)

        col_spacings = [max(map(len, col)) for col in repr_mat.list_cols()]

        new_mat = [[ele.rjust(padding) for ele, padding in zip(row, col_spacings)]
                                       for row in repr_mat.list_rows()]

        rep = ['[' + ', '.join(row) + ']' for row in new_mat]

        rep = [str(rep[0])] + \
                [" "*8 + str(row) for row in rep[1:]]

        return "Matrix([" + ",\n".join(rep) + "])"


    def __iter__(self):
        """
        This method LINEARIZES the elements of the matrix, which is
        useful because if you have a column/row vector you can
        iterate over it directly or turn it into a list.
        Syntax:
        for element in MyMatrix:
            do_something(element)

        for element in MyVector:
            do_something(element)

        >>> list(MyMatrix)
        [a11, a12, a13, ..., a1n, a21, a22, ..., ..., amn]
        >>> list(MyVector)
        [a1, a2, a3, ..., an]
        """
        # TODO: Test out matrices being iterable
        return (element for row in self.mat for element in row)

    def __len__(self):
        """
        Defines the output of the len function on a matrix.
        'len(self)' returns the number of elements in the matrix.
        Syntax: n = len(M)
        """
        return self.m * self.n

    def __eq__(self, other):
        """
        Tests the equality of two matrices.
        Syntax: A == B
        """
        if not isinstance(other, Matrix) or self.size != other.size:
            return False
        else:
            return self.mat == other.mat


    def __round__(self, ndigits=0):
        """
        Allows you to use the builtin round() function on a matrix.
        Syntax: round(M, 4)
        # rounds each element in matrix M to 4 decimal places
        """
        return self.under(lambda ele: round(ele, ndigits))

    def __neg__(self):
        """
        Returns a matrix where each element is the opposite of the corresponding element
        in self.
        Syntax: B = -A
        """
        return self.under(lambda ele: -ele)

    def __pos__(self):
        """
        Returns a matrix where each element is the absolute value of the corresponding
        element in self.
        Syntax: B = +A
        """
        return self.under(abs)

    def __add__(self, other):
        """
        Defines the + operator between matrices. Matrices must be the same size.
        Syntax: C = A + B
        """
        return ewise(self, other, func=operator.add)

    def __sub__(self, other):
        """
        Defines the - operator between matrices. Matrices must be the same size.
        Syntax: C = A - B
        """
        return self + -other

    def __mul__(self, other):
        """
        Defines the * operator for element-wise multiplication between matrices.
        Syntax: C = A * B
        """
        return ewise(self, other, func=operator.mul)

    def __rmul__(self, scalar):
        """
        Allows matrices to be multiplied by scalars on the LEFT side of the *.
        Syntax: B = 12 * A
        """
        if not isinstance(scalar, Matrix):
            return self.under(lambda ele: scalar * ele)
        else:
            raise TypeError("Scalar multiplication will intentionally fail if " +
                            "performed on another Matrix.")

    def __matmul__(self, other):
        """
        Defines the @ operator between matrices.
        Performs matrix multiplication.
        Syntax: C = A @ B
        """
        if not isinstance(other, Matrix):
            msg = "Cannot matmul (@) a matrix with the non-matrix instance: {}"
            raise TypeError(msg.format(repr(other)))
        if self.n == other.m: # If inside dimensions agree,

            new_m = self.m
            new_n = other.n

            # We are doing the calculation: self @ other
            new_mat = [[list_dot_product(row, col)
                        for col in other.list_cols()]
                            for row in self.list_rows()]
            return Matrix(new_mat)
        else:
            raise MismatchedMatrixSize("Matrix multiplication could not be " +
                                       "performed on matrices with different" +
                                       " interior dimensions.")

    def __pow__(self, exp):
        """
        Defines the ** operator between matrices.
        Performs matrix exponentiation where A ** 3 == A @ A @ A.
        """
        if int(exp) != exp:
            raise TypeError("Matrix exponentiation is not defined for non-integer exponents.")
        if self.m != self.n:
            raise TypeError("Cannot perform exponentiation on non-square matrix.")

        if exp > 0:
            factors = (self for _ in range(exp))
            return reduce(lambda A, B: A @ B, factors)
        elif exp < 0:
            return inv(self) ** abs(exp)
        else:
            return I(self.m)


    def matrix_of_cofactors(self):
        return checkerboard(Matrix([[det(self.minor(i, j))    # take determinant of each element's minor
                    for j, ele in enumerate(row)]
                        for i, row in enumerate(self.mat)]))


    def minor(self, ith, jth):
        """
        Returns a matrix made of all the elements of self EXCEPT those from
        row i and column j.
        Syntax: mnr = A.minor(3,2)
        """
        return Matrix([[ele for j, ele in enumerate(row) if j != jth]
                            for i, row in enumerate(self.mat) if i != ith])


    def reshape(self, new_size):
        """
        Allows the size of a matrix to be changed after the matrix is instantiated.
        Particularly useful if you want to linearize a matrix, perform an operation
        on each element, and then return it to its original size.
        """
        m, n = new_size
        if m * n != self.m * self.n:
            raise MatrixError("Reshape could not complete because new matrix size " +
                            "would be incompatible with current matrix size.")
        else:
            self.m, self.n = new_size
            expanded = [iter(self)] * self.n
            return Matrix(list(map(list, zip(*expanded))))


    ##### Alternate Constructors #####

    @classmethod
    def from_list_vector(self, L):
        """
        Creates a COLUMN vector from a list of values.
        """
        return Matrix([L]).T

    ##### Quaternion Extension #####

    @classmethod
    def from_quaternion(cls, r, i, j, k):
        """
        Generates a matrix representation of the given quaternion.
        Syntax: Q = Matrix.from_quaternion(1, 2, 3, 4)
        """
        mat = [[r, -i, -j, -k],
               [i,  r, -k,  j],
               [j,  k,  r, -i],
               [k, -j,  i,  r]]
        return cls(mat) # returns a Matrix (or subclass) OBJECT

    def as_quaternion(self):
        """
        If the matrix can be represented as a quaternion, this method
        returns a list of the real, i, j, and k components of the
        quaternion.

        Syntax:
        >>> Matrix.from_quaternion(1,2,3,4).as_quaternion()
        [1,2,3,4]
        """
        if self.size != (4, 4):
            raise InvalidMatrixFormat("Cannot represent matrix as quaternion. " +
                "Matrix was not 4x4.")
        elif self != Matrix.from_quaternion(*list(self[:, 0])):
            raise InvalidMatrixFormat("Cannot represent matrix as quaternion. " +
                "Matrix did not match matrix-quaternion template.")
        else:
            return list(self[:, 0])


##### Functions of the Matrix Module #####

def ewise(M1, M2, func):
    """
    Performs an element-wise calculation given two matrices and a
    TWO-VARIABLE function. Returns the result.
    Syntax: C = Matrix.ewise(A, B, func=math.atan2)
    """
    if M1.size == M2.size:
        new_mat = [[func(*pair) for pair in zip(M1, M2)]]
        return Matrix(new_mat).reshape(M1.size)
    else:
        raise MismatchedMatrixSize("This operation is only supported for matrices of " +
                                   "the same size. Did you mean to use @ instead of *?")


def list_dot_product(L1, L2):
    """
    Returns the dot product of two LISTS.
    Syntax: res = list_dot_product([1, 2, 3], [4, 5, 6])
    """
    # return sum(map(mul, L1, L2))
    return sum(ele1 * ele2 for ele1, ele2 in zip(L1, L2))


def format_slice(slc, overwriting_matrix, kind="col"):
    if type(slc) is not slice:
        raise TypeError("format_slice only accepts slice objects as it's " +
            "first argument. ({} provided)".format(type(slc)))

    if kind == "col":
        alt_dim = overwriting_matrix.n
    elif kind == "row":
        alt_dim = overwriting_matrix.m
    else:
        raise TypeError("Keyword argument 'kind' only takes 'col' or 'row'.")

    start = 0 if slc.start is None else slc.start
    stop = alt_dim if slc.stop is None else slc.stop

    start = alt_dim + start if start < 0 else start
    stop = alt_dim + stop if stop < 0 else stop

    return slice(start, stop, None)


def I(n):
    """
    Returns the multiplicative identity matrix of size 'nxn'.
    Syntax: iden = I(5)
    """
    return Matrix([[1 if i == j else 0
            for j in range(n)]
                for i in range(n)])

def Z(m, n=None):
    """
    Returns the additive identity matrix of size 'mxn'.
    Syntax:
    z_5x5 = Z(5)
    z_1x99 = Z(1, 99)
    """
    # TODO: Test Z function
    if n == None:
        n = m
    return Matrix([[0 for j in range(n)] for i in range(m)])

def derivative_matrix(degree):
    """
    Returns a derivative operator for polynomials of a certain degree.

    derivative_matrix(n) @ P_n == P_n'
    where:
        P_n is an n-degree polynomial represented as a vector,
        P_n' is the derivative of P_n
    """
    if degree == 0:
        return Matrix([[0]])
    elif degree > 0:
        return Matrix([[(row + 1) if (row + 1) == col else 0
                    for col in range(degree + 1)]
                        for row in range(degree + 1)])
    else:
        raise MatrixError("Do not pass a negative value as the degree " +
            "of the derivative operator D.")


def ref(M):
    """
    Returns the row echelon form of the matrix M.
    """
    # Copy M to N
    N = Matrix(M.mat.copy())

    print("\nREF of", N, sep="\n")

    # Base case
    if N.m == 1: # N is an mxn matrix
        if N[0,0] in [0, 1]:
            return N
        else:
            return 1/N[0,0] * N

    # if col 0 looks like [0,0,0,...]^T or [1,0,0,...]^T
    if N[0,0] in [1, 0] and all(map(lambda e: e == 0, N[1: , 0])):
        N[1: , 1: ] = ref(N[1: , 1: ])
        return N

    if N[0,0] == 0:
        N.mat[-1] = N.mat.pop(0) # send top row to bottom
        return ref(N)
    else:
        # Give first row a leading 1
        if N[0,0] != 1:
            N[0] = 1/N[0,0] * N[0]
        
        # Define a row replacement function
        def replace(row):
            print("\nreplace(", repr(row), ")")
            print("row.mat:", row.mat)
            print("N.mat:", N.mat)
            print("N[0]:", N[0])
            print("N[0].mat:", N[0].mat)
            print("row[0,0]:", row[0,0])
            return row - row[0,0]*N[0] \
                    if row[0,0] != 0 else row # avoid unnecessary division

        head = N[0]
        print("\nHead:", repr(head))
        tail = list(map(lambda m: m.mat[0], map(replace, N[1:].row_vecs())))
        print("Tail:", tail)
        return ref(Matrix(head + tail))


def checkerboard(M):
    """
    Multiplies the given matrix by a matrix of alternating +1s and -1s.
    Example:
    >>> M = Matrix([[1, 2, 3],
                    [5, 6, 7],
                    [7, 8, 9]])
    >>> checkerboard(M)
    Matrix([[ 1, -2,  3],
            [-4,  5, -6],
            [ 7, -8,  9]])
    """
    return Matrix([[ele * (-1)**(i + j)
            for j, ele in enumerate(row)]
                for i, row in enumerate(M.list_rows())])

def det(M):
    """
    Returns the determinant of the given matrix.
    Syntax: x = det(M)
    """
    if M.m != M.n:
        raise MatrixError("Determinant is not defined for non-square matrices." +
                            "Was given matrix of size {}x{}.".format(M.m, M.n))
    elif M.size == (1, 1):
        return M[0,0]
    elif M.size == (3, 3):
        return M[0,0]*(M[1,1]*M[2,2] - M[2,1]*M[1,2]) - \
               M[0,1]*(M[1,0]*M[2,2] - M[2,0]*M[1,2]) + \
               M[0,2]*(M[1,0]*M[2,1] - M[2,0]*M[1,1])
    else:
        # 1) For each element in the first row of the checkerboard (all are either +/-1):
        #       a) Multiply that element by the determinant of that element's minor.
        #       b) Add that result to the running total.
        # 2) Return the total.
        return sum(row_1_ele * det(M.minor(0, j))
                for j, row_1_ele in enumerate(checkerboard(M)[0]))

def adj(M):
    return M.matrix_of_cofactors().T

def inv(M):
    """
    Returns the multiplicative inverse of the given matrix.
    Syntax: A_inv = inv(A)
    """
    Det = det(M)
    if Det == 0:
        raise MatrixError("This matrix is not invertible becuase it has a zero determinant.")
    else:
        return 1/Det * adj(M)

def characteristic_polynomial(M):
    if M.m != M.n:
        raise MatrixError("Eigenvalues cannot be found for non-square matrices.")
    else:
        lambda_ = Polynomial("x")
        L = I(M.n).under(lambda e: e * lambda_)
        return det(L - M)

def eigenvalues(M):
    """
    Returns the eigenvalues of the given matrix.
    Syntax: eigs = eigenvalues(M)
    """
    return roots(characteristic_polynomial(M))


class MatrixError(Exception):
    """
    A general matrix exception class.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

class MismatchedMatrixSize(MatrixError):
    """
    Exception class whose instances can be called when an element-wise
    calculation is called on two matrices of different sizes.
    """
    pass

class InvalidMatrixFormat(MatrixError):
    """
    No docs here...
    """
    pass


if __name__ == '__main__':
    M = Matrix([[ 0, 1, 2, 3, 4, 5, 6, 7],
                [ 9, 8, 7, 6, 5, 4, 3, 2],
                [-0,-1,-2,-3,-4,-5,-6,-7],
                [-9,-8,-7,-6,-5,-4,-3,-2],
                [-0, 1,-2, 3,-4, 5,-6, 7],
                [-9, 8,-7, 6,-5, 4,-3, 2]])

    # print(M)
    # print()

    # out = M[1:3, 1:3]
    # print(out)
    # print(type(out))

    # M[-4:-1, -3:] = Matrix([[9999, 9999, 8888],
    #                       [9999, 9999, 8888],
    #                       [9999, 9999, 8888]])
    # print(M)
    
    print(ref(M))


    print()





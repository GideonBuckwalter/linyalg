import operator
from functools import reduce
from poly import *

__author__ = "Gideon Buckwalter"

"""
Author: Gideon Buckwalter
Email: gideon.buckwalter@gmail.com
GitHub: @GideonBuckwalter
Creation Date: 29 June 2016
Version: Python 3.5.2
"""

class Matrix(object):
    def __init__(self, mat=[[]]):
        """
        Pass a two-dimensional list containing the matrix data.
        A size must be provided if you populate the matrix with
        some sort of iterable (eg. 2D generator) (<- bad idea!).
        """
        self.m = len(mat)
        self.n = len(mat[0])
        self.mat = mat

    def size(self):
        """
        Returns a tuple containing the size, n x m, of the matrix.
        """
        return (self.m, self.n)

    def col(self, col):
        """
        Returns the column vector at the specified index.
        """
        return [row[col] for row in self.mat]

    def row(self, row):
        """
        Returns the row vector at the specified index.
        """
        return self.mat[row]

    def rows(self):
        """
        Returns an iterable of all the rows in the matrix.
        """
        return iter(self.mat)

    def cols(self):
        """
        Returns and iterable of all the columns in the matrix.
        """
        return (self.T).rows()

    def submatrix(self, pair1, pair2):
        """
        Returns the submatrix that goes from pair1 to pair2
        :param pair1: Tuple of starting indices (m1, n1)
        :param pair2: Tuple of ending indices (m2, n2)
        :return: Matrix
        """
        m1, n1 = pair1
        m2, n2 = pair2
        return Matrix([row[n1:n2] for row in self.mat[m1:m2]])

    def under(self, func):
        """
        A.under(math.cos) will return a copy of Matrix A where each element
        has been replaced with the cos of that element. Matrix.under only
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

        col_spacings = [max(map(len, col)) for col in str_mat.cols()]

        new_mat = [[ele.rjust(padding) for ele, padding in zip(row, col_spacings)] for row in str_mat.rows()]

        rep = ['[' + ', '.join(row) + ']' for row in new_mat] + \
              [str(self.m) + "x" + str(self.n)]
        return "\n".join(rep)

    def __repr__(self):
        rep = [str(self.mat[0])] + \
                [" "*7 + str(row) for row in self.mat[1:]]
        return "Matrix(" + "\n".join(rep) + ")"

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
                m1, m2 = i.start, i.stop
                if type(j) is slice:
                    n1, n2 = j.start, j.stop
                elif type(j) is int:
                    n1, n2 = j, j+1
                else:
                    msg = "Invalid matrix index argument type: {}, {}"
                    raise TypeError(msg.format(type(i), type(j)))

                return self.submatrix((m1,n1), (m2,n2))

            elif type(i) is int:
                if type(j) is int:
                    return self.mat[i][j]
                elif type(j) is slice:
                    return Matrix([self.mat[i][j]])
                else:
                    msg = "Invalid matrix index argument type: {}, {}"
                    raise TypeError(msg.format(type(i), type(j)))
        
        elif type(key) in (int, slice):
            return Matrix([self.mat[key]])
        else:
            raise TypeError("Invalid matrix index type: {}".format(type(inp)))

    def __setitem__(self, key, value):
        """
        Allows mutable bracket assignments.
        Syntax: M[1] = [3, 4, 5]
                M[1][2] = 100
        """
        self.mat[key] = value

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
        return self.mat == other.mat

    def __round__(self, ndigits=0):
        """
        Allows you to use the builtin round() function on a matrix.
        Syntax: round(M, 4)
        # rounds each element in matrix M to 4 decimal places
        """
        round_ndigits = lambda x: round(x, ndigits)
        return self.under(round_ndigits)

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
        Allows matrices to be multiplied by scalars on the left side of the *.
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
        if self.n == other.m: # If inside dimensions agree,

            new_m = self.m
            new_n = other.n

            # We are doing the calculation: self @ other
            new_mat = [[dot_product(row, col)
                        for col in other.cols()]
                            for row in self.rows()]
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



    @property
    def T(self):
        """
        Returns the transpose of the matrix. Turns all rows into columns and all columns
        into rows.
        Syntax: B = A.T
        """
        return Matrix([self.col(i) for i in range(self.n)])

    # Elementary Row Operations
    #       These are mutable since it it likely that you will want to
    #       make a copy and then completely transform the matrix by
    #       row reducing.
    def swap_rows(self, Ra, Rb):
        self[Ra], self[Rb] = self[Rb], self[Ra]

    def scale_row(self, R, by):
        self[R] = [by * ele for ele in self[R]]

    def to_Ra_add_cRb(self, Ra, c, Rb):
        """
        Mutates self by adding c * row[Rb] to row[Ra].
        Syntax:
        M.to_Ra_add_cRb(Ra=3, c=-6.32, Rb=0)
        """
        self[Ra] = [Ra_ele + c * Rb_ele
                        for Ra_ele, Rb_ele in
                        zip(self[Ra], self[Rb])]


    def matrix_of_cofactors(self):
        return checkerboard(Matrix([[det(self.minor(i, j))    # take determinant of each element's minor
                    for j, ele in enumerate(row)]
                        for i, row in enumerate(self.mat)]))


    def minor(self, i, j):
        """
        Returns a matrix made of all the elements of self EXCEPT those from
        row i and column j.
        Syntax: mnr = A.minor(3,2)
        """
        return Matrix([[self[row][col]
                    for col in range(self.n) if col != j]
                        for row in range(self.m) if row != i])


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
            return Matrix(list(map(lambda tup: list(tup), zip(*expanded))))

    def zeros(self):
        return self.under(lambda ele: ele == 0)


    ##### Alternate Constructors #####

    @classmethod
    def from_list_vector(self, L):
        """
        Creates a COLUMN vector from a list of values.
        """
        return Matrix([L]).T

    def as_list_vector(self):
        """
        Returns a list representation of a column vector OR row vector.
        Note: this is NOT a reversible function for ROW vectors.
        """
        if self.m == 1: # must be a row vector
            return self.row(0)
        if self.n == 1: # must be a column vector
            return self.T.as_list_vector()

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
        if self.size() != (4, 4):
            raise InvalidMatrixFormat("Cannot represent matrix as quaternion. " +
                "Matrix was not 4x4.")
        elif self != Matrix.from_quaternion(*self.col(0)):
            raise InvalidMatrixFormat("Cannot represent matrix as quaternion. " +
                "Matrix did not match matrix-quaternion template.")
        else:
            return self.col(0)


##### Functions of the Matrix Module #####

def ewise(M1, M2, func):
    """
    Performs an element-wise calculation given two matrices and a
    TWO-VARIABLE function. Returns the result.
    Syntax: C = Matrix.ewise(A, B, func=math.atan2)
    """
    if M1.size() == M2.size():
        new_mat = [[func(*pair) for pair in zip(M1, M2)]]
        return Matrix(new_mat).reshape(M1.size())
    else:
        raise MismatchedMatrixSize("This operation is only supported for matrices of " +
                                   "the same size. Did you mean to use @ instead of *?")


def dot_product(vec1, vec2):
    """
    Returns the dot product of two LISTS.
    Syntax: res = dot_product([1, 2, 3], [4, 5, 6])
    """
    return sum(ele1 * ele2 for ele1, ele2 in zip(vec1, vec2))


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

def row_swap(M, i1, i2):
    """Should this even be here? Maybe make it a mutable method?"""
    copy = M
    copy[i1], copy[i2] = M[i2], M[i1]
    return copy



# function retval = rref (m, dummy) 
def ref(M):
#      if (nargin < 1) 
#         usage ("rref (x)"); 
#      endif 

# # Bring the pivot to the top row 
    for i in range(M.m):
        if M[i][0] == max(M.col(0)):
            M[i], M[0] = M[0], M[i]
#      for i=1:rows(m) 
#        if(m(i,1)==max(m(:,1))) 
#          scratch=m(i,:); 
#          m(i,:)=m(1,:); 
#          m(1,:)=scratch; 

#        endif    
#      endfor 
    if M[0][0] != 0:
        M[0] = [ele/M[0][0] for ele in M[0]]
        for i in range(1,M.m):
            if M[i][0] != 0:
                M[i] = [ele/M[i][0] - top_ele
                            for ele, top_ele in zip(M[i], M[0])]
#      if(m(1,1)!=0) 
#        m(1,:)=m(1,:)/m(1,1); 

#        for i=2:rows(m) 

#          if(m(i,1)!=0) 

#            m(i,:)=(m(i,:)/m(i,1))-m(1,:); 

#          endif 
#        endfor 
    else:
        if any(M[0]):
            q = 1
            while M[0][q] == 0:
                q += 1
            M[0] = [ele/M[0][q] for ele in M[0]]
#      else 
#          if(any(m(1,:))) 
#            q=2; 
#            while(m(1,q)==0) 
#              q++; 
#            endwhile 
#            m(1,:)=m(1,:)/m(1,q); 
#          endif 
#      endif 
    if M.n > 1 and M.n > 1:
        M[1 : M.m-1][1 : M.n-1] = rref(M[2 : ])
#      if(columns(m)>1 && rows(m)>1) 
#         m(2:rows(m),2:columns(m)) = rref( m(2:rows(m),2:columns(m)),1 );    
#      endif 


# # Swell, now we're row reduced.  Lets get to rref 

#      if((rows(m)>=2) && (columns(m) >=2) && (nargin ==1)) 

#      for i=0:rows(m)-1 
#        if( any( m(rows(m)-i,:) ) ) 

#          q=1; 
#          while(m(rows(m)-i,q)==0) 
#            q++; 
#          endwhile 
#          for j=1:rows(m)-i-1 
#            m(j,:)=m(j,:)-(m(rows(m)-i,:)*m(j,q)); 
#          endfor 

#        endif 
#      endfor 

#      endif 

#      retval = m; 

# endfunction



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
                for i, row in enumerate(M.rows())], size=(M.m, M.n))

def det(M):
    """
    Returns the determinant of the given matrix.
    Syntax: x = det(M)
    """
    if M.m != M.n:
        raise MatrixError("Determinant is not defined for non-square matrices." +
                            "Was given matrix of size {}x{}.".format(M.m, M.n))
    elif M.size() == (1, 1):
        return M[0][0]
    elif M.size() == (3, 3):
        return M[0][0]*(M[1][1]*M[2][2] - M[2][1]*M[1][2]) - \
               M[0][1]*(M[1][0]*M[2][2] - M[2][0]*M[1][2]) + \
               M[0][2]*(M[1][0]*M[2][1] - M[2][0]*M[1][1])
    else:
        # 1) For each element in the first row of the checkerboard:
        #       a) Multiply that element by the determinant of that element's minor.
        #       b) Add that result to the running total.
        # 2) Return the total.
        return sum(row_1_ele * det(M.minor(0, j))
                for j, row_1_ele in enumerate(checkerboard(M).row(0)))

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
    M = Matrix([[0,1,2,3,4,5,6,7],
                [9,8,7,6,5,4,3,2],
                [-0,-1,-2,-3,-4,-5,-6,-7],
                [-9,-8,-7,-6,-5,-4,-3,-2],
                [-0,1,-2,3,-4,5,-6,7],
                [-9,8,-7,6,-5,4,-3,2]])
    print(M)
    print()

    out = M[:, :]
    print(out)
    print(type(out))
    


    print()





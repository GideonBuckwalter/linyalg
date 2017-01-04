import pytest
from .matrix import *
from .matrix import format_slice, checkerboard, ewise, list_dot_product
import copy, itertools
from fractions import Fraction
from random import randint, uniform, choice

# Define random-matrix generators

def randmat(randfunc, m=None, n=None, rng=100):
    if m is None: m = randint(1,10)
    if n is None: n = randint(1,10)
    return Matrix([[randfunc(-rng, rng) for col in range(n)] for row in range(m)])

def intmat(m=None, n=None, rng=10):
    return randmat(randint, m, n, rng)

def realmat(m=None, n=None, rng=1):
    return randmat(uniform, m, n, rng)

def sparcemat(m=None, n=None, rng=10, sparcity=0.6):
    def picker(low, high):
        choice([randint(low, high)]*int((1-sparcity)*100) + [0]*int(sparcity*100))
    return randmat(picker, m, n, rng)



test_matrices = []

A_4x4 = Matrix([[ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16]])
test_matrices.append(A_4x4)

# A Singular matrix
Sing_4x4 = Matrix([[1, 5,-3, 3],
                   [3, 2, 5,10],
                   [5, 1,-6, 0],
                   [6, 8, 0,14]])
test_matrices.append(Sing_4x4)

# A symmetric matrix
Sym_4x4 = Matrix([[1, 2, 3, 4],
                  [2, 5, 6, 7],
                  [3, 6, 8, 9],
                  [4, 7, 9,10]])
test_matrices.append(Sym_4x4)

Cons_4x4 = Matrix([[ 0,-9, 8, 7],
                   [12,13,14,15],
                   [-3, 5, 2, 1],
                   [ 6, 0, 6, 1]])
test_matrices.append(Cons_4x4)

A_3x3 = Matrix([[1,2,3],
                [6,5,4],
                [7,8,9]])
test_matrices.append(A_3x3)
Sing_3x3 = Matrix([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
test_matrices.append(Sing_3x3)

A_3x2 = Matrix([[1, 2],
                [3, 4],
                [5, 6]])
test_matrices.append(A_3x2)

B_3x2 = Matrix([[ 7,  8],
                [ 9, 10],
                [11, 12]])
test_matrices.append(B_3x2)

A_2x2 = Matrix([[1,2],
                [3,4]])
test_matrices.append(A_2x2)

Sing_2x2 = Matrix([[1,2],
                   [4,8]])
test_matrices.append(Sing_2x2)

A_1x1 = Matrix([[9]])
test_matrices.append(A_1x1)


def test_mat_attribute():
    assert A_4x4.mat == [[ 1,  2,  3,  4],
                         [ 5,  6,  7,  8],
                         [ 9, 10, 11, 12],
                         [13, 14, 15, 16]]
    assert A_3x2.mat == [[1, 2],
                         [3, 4],
                         [5, 6]]
    assert A_1x1.mat == [[9]]

def test_list_rows():
    assert list(A_4x4.list_rows()) == [[ 1,  2,  3,  4],
                                       [ 5,  6,  7,  8],
                                       [ 9, 10, 11, 12],
                                       [13, 14, 15, 16]]
    assert list(A_3x2.list_rows()) == [[1, 2],
                                       [3, 4],
                                       [5, 6]]
    assert list(A_1x1.list_rows()) == [[9]]

def test_list_cols():
    assert list(A_4x4.list_cols()) == [[1,5, 9,13],
                                       [2,6,10,14],
                                       [3,7,11,15],
                                       [4,8,12,16]]
    assert list(A_3x2.list_cols()) == [[1,3,5],
                                       [2,4,6]]
    assert list(A_1x1.list_cols()) == [[9]]

def test_matrix_equality():
    for M in test_matrices:
        N = copy.deepcopy(M)
        O = copy.deepcopy(N)

        # Show they are not the same objects
        assert M is not N
        assert N is not O
        assert O is not M

        # Reflexivity
        assert M == M
        # Symmetry
        assert M == N and N == M
        # Transitivity
        assert M == N and N == O and M == O

def test_not_equal():
    for M, N in itertools.combinations(test_matrices, 2):
        assert M is not N
        assert M != N
        assert N != M

def test_transpose():
    assert A_4x4.T == Matrix([[1, 5, 9, 13],
                              [2, 6, 10, 14],
                              [3, 7, 11, 15],
                              [4, 8, 12, 16]])
    assert A_3x2.T == Matrix([[1, 3, 5],
                              [2, 4, 6]])
    assert Sym_4x4.T == Sym_4x4
    assert A_1x1.T == A_1x1

def test_getitem():
    assert A_4x4[0,0] == 1
    assert A_3x2[0,0] == 1
    assert A_1x1[0,0] == 9

    assert A_4x4[0] == A_4x4[0, :] == A_4x4[0:1] == Matrix([[1,2,3,4]])
    assert A_3x2[0] == A_3x2[0, :] == A_3x2[0:1] == Matrix([[1,2]])
    assert A_1x1[0] == A_1x1[0, :] == A_1x1[0:1] == A_1x1

    assert A_4x4[1:3] == Matrix([[5, 6, 7, 8],
                                 [9,10,11,12]])
    assert A_3x2[0:2] == Matrix([[1,2],
                                 [3,4]])
    assert A_1x1[0:1] == A_1x1

    assert A_4x4[1:3, :] == A_4x4[1:3]
    assert A_3x2[0:2, :] == A_3x2[0:2]
    assert A_1x1[0:1, :] == A_1x1[0:1]

    assert A_4x4[:,0] == Matrix([[1],[5],[9],[13]])
    assert A_3x2[:,0] == Matrix([[1],[3],[5]])
    assert A_1x1[:,0] == A_1x1

    assert A_4x4[:,:] == A_4x4[:] == A_4x4
    assert A_3x2[:,:] == A_3x2[:] == A_3x2
    assert A_1x1[:,:] == A_1x1[:] == A_1x1

    assert A_4x4[0:2, 0:2] == A_4x4[:2, :2] == Matrix([[1,2],
                                                       [5,6]])
    assert A_3x2[0:2, 0:2] == A_3x2[:2, :2] == A_3x2[0:2, 0:] == Matrix([[1,2],
                                                                         [3,4]])
    assert A_1x1[0:2, 0:2] == A_1x1[:2] == A_1x1[:2, :2] == A_1x1[0:, 0:] == A_1x1

def test_format_slice():

    class SliceGetter:
        def __getitem__(self, key):
            return key

    sg = SliceGetter()

    M = A_4x4

    slice_tests = (
        (sg[1:3], sg[1:3]),
        (sg[:3], sg[0:3]),
        (sg[1:], sg[1:4]),
        (sg[:], sg[0:4]),
        (sg[1:-1], sg[1:3]),
        (sg[-2:-1], sg[2:3])
        )

    for inp, ans in slice_tests:
        assert format_slice(inp, M, kind="col") == ans
        assert format_slice(inp, M, kind="row") == ans

def test_setitem():
    A44 = copy.deepcopy(A_4x4)
    A32 = copy.deepcopy(A_3x2)
    A11 = copy.deepcopy(A_1x1)

    A44[0,0] = -7
    A32[0,0] = -7
    A11[0,0] = -7

    assert A44 == Matrix([[-7,  2,  3,  4],
                          [ 5,  6,  7,  8],
                          [ 9, 10, 11, 12],
                          [13, 14, 15, 16]])
    assert A32 == Matrix([[-7, 2],
                          [ 3, 4],
                          [ 5, 6]])
    assert A11 == Matrix([[-7]])

    A44 = copy.deepcopy(A_4x4)
    A32 = copy.deepcopy(A_3x2)
    A11 = copy.deepcopy(A_1x1)

    A44[0:2, 0:2] = Matrix([[-7,-7],
                            [-7,-7]])
    A32[0:2, 0:2] = Matrix([[-7,-7],
                            [-7,-7]])
    A11[0:1, 0:1] = Matrix([[-7]])

    assert A44 == Matrix([[-7,-7,  3,  4],
                          [-7,-7,  7,  8],
                          [ 9, 10,11, 12],
                          [13, 14,15, 16]])
    assert A32 == Matrix([[-7,-7],
                          [-7,-7],
                          [ 5, 6]])
    assert A11 == Matrix([[-7]])

    A44 = copy.deepcopy(A_4x4)
    A32 = copy.deepcopy(A_3x2)
    A11 = copy.deepcopy(A_1x1)

    A44[2] = Matrix([[-7,-7,-7,-7]])
    A32[0] = Matrix([[-7,-7]])
    A11[0] = Matrix([[-7]])

    assert A44 == Matrix([[ 1, 2, 3, 4],
                          [ 5, 6, 7, 8],
                          [-7,-7,-7,-7],
                          [13,14,15,16]])
    assert A32 == Matrix([[-7,-7],
                          [ 3, 4],
                          [ 5, 6]])
    assert A11 == Matrix([[-7]])

    A44 = copy.deepcopy(A_4x4)
    A32 = copy.deepcopy(A_3x2)
    A11 = copy.deepcopy(A_1x1)

    A44[0:2] = Matrix([[-7,-7,-7,-7],
                       [-7,-7,-7,-7]])
    A32[0:2] = Matrix([[-7,-7],
                       [-7,-7]])
    A11[0:1] = Matrix([[-7]])

    assert A44 == Matrix([[-7,-7, -7, -7],
                          [-7,-7, -7, -7],
                          [ 9, 10,11, 12],
                          [13, 14,15, 16]])
    assert A32 == Matrix([[-7,-7],
                          [-7,-7],
                          [ 5, 6]])
    assert A11 == Matrix([[-7]])

    A44 = copy.deepcopy(A_4x4)
    A32 = copy.deepcopy(A_3x2)
    A11 = copy.deepcopy(A_1x1)

    A44[:2, 2] = Matrix([[-7],
                         [-7]])
    A32[:2, 1] = Matrix([[-7],
                         [-7]])
    A11[:1, 0] = Matrix([[-7]])

    assert A44 == Matrix([[ 1, 2,-7, 4],
                          [ 5, 6,-7, 8],
                          [ 9,10,11,12],
                          [13,14,15,16]])
    assert A32 == Matrix([[1,-7],
                          [3,-7],
                          [5, 6]])
    assert A11 == Matrix([[-7]])

    A44 = copy.deepcopy(A_4x4)
    # Test row swapping
    A44[0], A44[1] = A44[1], A44[0]
    assert A44 == Matrix([[ 5,  6,  7,  8],
                          [ 1,  2,  3,  4],
                          [ 9, 10, 11, 12],
                          [13, 14, 15, 16]])


def test_row_vecs():
    assert list(A_4x4.row_vecs()) == [Matrix([[ 1,  2,  3,  4]]),
                                      Matrix([[ 5,  6,  7,  8]]),
                                      Matrix([[ 9, 10, 11, 12]]),
                                      Matrix([[13, 14, 15, 16]])]
    assert list(A_3x2.row_vecs()) == [Matrix([[1,2]]),
                                      Matrix([[3,4]]),
                                      Matrix([[5,6]])]
    assert list(A_1x1.row_vecs()) == [Matrix([[9]])]

def test_col_vecs():
    assert list(A_4x4.col_vecs()) == [Matrix([[1],[5], [9],[13]]),
                                      Matrix([[2],[6],[10],[14]]),
                                      Matrix([[3],[7],[11],[15]]),
                                      Matrix([[4],[8],[12],[16]])]

    assert list(A_3x2.col_vecs()) == [Matrix([[1],[3],[5]]),
                                      Matrix([[2],[4],[6]])]

    assert list(A_1x1.col_vecs()) == [Matrix([[9]])]

def test_under():
    assert A_4x4.under(lambda x: -x) == Matrix([[ -1, -2, -3, -4],
                                                [ -5, -6, -7, -8],
                                                [ -9,-10,-11,-12],
                                                [-13,-14,-15,-16]])
    assert A_3x2.under(str) == Matrix([["1", "2"],
                                       ["3", "4"],
                                       ["5", "6"]])
    assert A_1x1.under(lambda x: x - 9) == Matrix([[0]])


def test_str():
    M = Matrix([[134,-2,  531],
                [  4, 5,215.6],
                [ -7, 8,    9]])

    ans = '\n'.join(["[134, -2,   531]",
                     "[  4,  5, 215.6]",
                     "[ -7,  8,     9]",
                     "3x3"])
    assert str(M) == ans

def test_repr():
    M = Matrix([[134,-2,  531],
                [  4, 5,215.6],
                [ -7, 8,    9]])

    ans = '\n'.join(["Matrix([[134, -2,   531],",
                     "        [  4,  5, 215.6],",
                     "        [ -7,  8,     9]])"])
    assert repr(M) == ans

def test_iter():
    ans = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    assert list(A_4x4) == list(iter(A_4x4)) == ans

    ans = [1,2,3,4,5,6]
    assert list(A_3x2) == ans

    row_vec = Matrix([[1,2,3,4]])
    col_vec = row_vec.T
    assert list(row_vec) == list(col_vec) == [1,2,3,4]

def test_len():
    assert len(A_4x4) == 16
    assert len(A_3x2) == 6
    assert len(A_1x1) == 1

def test_round():
    M = Matrix([[ 0.333,   0.555, 0.00999],
                [12.333, -31.555, 7.00999]])
    ans = Matrix([[ 0.33,   0.56, 0.01],
                  [12.33, -31.55, 7.01]])
    #                          ^odd case
    assert round(M, ndigits=2) == ans

def test_neg():
    ans = Matrix([[-1, -5,  3,  -3],
                  [-3, -2, -5, -10],
                  [-5, -1,  6,   0],
                  [-6, -8,  0, -14]])
    assert -Sing_4x4 == ans

def test_pos():
    ans = Matrix([[1, 5, 3, 3],
                  [3, 2, 5,10],
                  [5, 1, 6, 0],
                  [6, 8, 0,14]])
    assert +Sing_4x4 == ans


def test_addition():
    ans = Matrix([[ 2, 7,12,17],
                  [ 7,12,17,22],
                  [12,17,22,27],
                  [17,22,27,32]])
    assert A_4x4 + A_4x4.T == ans

def test_subtraction():
    ans = Matrix([[ 1-7,  2-8],
                  [ 3-9, 4-10],
                  [5-11, 6-12]])
    assert A_3x2 - B_3x2 == A_3x2 + -B_3x2 == ans

def test_elementwise_multiplication():
    v1 = Matrix([[1,2,3,4]]).T
    v2 = Matrix([[0,9,8,7]]).T

    assert v1 * v2 == Matrix([[0,18,24,28]]).T

    ans = Matrix([[ 1*7,  2*8],
                  [ 3*9, 4*10],
                  [5*11, 6*12]])
    assert A_3x2 * B_3x2 == ans

def test_scalar_multiplication():
    ans = Matrix([[ 3,  6,  9, 12],
                  [15, 18, 21, 24],
                  [27, 30, 33, 36],
                  [39, 42, 45, 48]])
    assert 3 * A_4x4 == ans


def test_matrix_multiplication():
    ans = Matrix([[ 46, 44,-11, 79],
                  [106,108,-27,187],
                  [166,172,-43,295],
                  [226,236,-59,403]])

    assert A_4x4 @ Sing_4x4 == ans

    v1 = Matrix([[1,2,3,4]]).T
    v2 = Matrix([[0,9,8,7]]).T
    assert sum(v1 * v2) == (v1.T @ v2)[0,0]

    assert A_1x1 * A_1x1 == Matrix([[81]])

def test_matrix_exponentiation():
    ans = Matrix([[ 897, 967,-117,1747],
                  [1275,2001, 412,3688],
                  [ 321, 158,  15, 494],
                  [2462,3260, 316,6038]])
    assert Sing_4x4 ** 3 == ans

    S = Sym_4x4
    assert S ** -4 == inv(S) @ inv(S) @ inv(S) @ inv(S)
    assert S ** 0 == I(S.m)


def test_minor():
    assert A_4x4.minor(0,0) == A_4x4[1:, 1:]
    assert A_4x4.minor(3,3) == A_4x4[:-1, :-1]

    assert A_3x2.minor(0,1) == A_3x2[1:, 0]

    ans = Matrix([[ 1,  2,  4],
                  [ 5,  6,  8],
                  [13, 14, 16]])
    assert A_4x4.minor(2,2) == ans

    with pytest.raises(Exception):
        A_1x1.minor(0,0)

def test_checkerboard():
    ans = Matrix([[  1, -2,  3, -4],
                  [ -5,  6, -7,  8],
                  [  9,-10, 11,-12],
                  [-13, 14,-15, 16]])
    assert checkerboard(A_4x4) == ans

def test_Z():
    assert Z(1) == Matrix([[0]])
    assert Z(2) == 0 * A_2x2
    assert Z(3,2) == 0 * A_3x2

def test_I():
    assert I(1) == Matrix([[1]])
    assert I(3) == Matrix([[1,0,0],
                           [0,1,0],
                           [0,0,1]])
    assert A_4x4 @ I(4) == I(4) @ A_4x4 == A_4x4

def test_matrix_of_cofactors():
    ans = Matrix([[4,3],
                  [2,1]])
    assert A_2x2.matrix_of_cofactors() == checkerboard(ans)

    ans = Matrix([[45-32,54-28,48-35],
                  [18-24, 9-21, 8-14],
                  [ 8-15, 4-18, 5-12]])
    assert A_3x3.matrix_of_cofactors() == checkerboard(ans)

    assert A_4x4.matrix_of_cofactors().matrix_of_cofactors() == Z(4)

def test_reshape():
    ans = Matrix([[1, 2, 3, 4, 5, 6, 7, 8],
                  [9,10,11,12,13,14,15,16]])
    cpy = copy.deepcopy(A_4x4)
    assert cpy.reshape([2,8]) == ans

    assert A_3x2.reshape([3,2]) == A_3x2

def test_augmented():
    with pytest.raises(TypeError):
        A_4x4.augmented(A_3x3)
    assert A_3x3.augmented(I(3)) == Matrix([[1,2,3,1,0,0],
                                            [6,5,4,0,1,0],
                                            [7,8,9,0,0,1]])

def test_with_coords():
    ans = Matrix([[("1",0,0),("2",0,1),("3",0,2)],
                  [("6",1,0),("5",1,1),("4",1,2)],
                  [("7",2,0),("8",2,1),("9",2,2)]])
    assert A_3x3.under(str).with_coords() == ans


# Test class methods

def test_from_list_vector():
    A = Matrix.from_list_vector([1,2,3,4,5])
    ans = Matrix([[1,2,3,4,5]]).T
    assert A == ans

def test_from_quaternion():
    one = Matrix.from_quaternion(1, 0, 0, 0)
    i   = Matrix.from_quaternion(0, 1, 0, 0)
    j   = Matrix.from_quaternion(0, 0, 1, 0)
    k   = Matrix.from_quaternion(0, 0, 0, 1)

    # Test quaternion to matrix conversion
    assert one == I(4)

    # Test quaternion multiplication properties
    assert i @ i == -one
    assert j @ j == -one
    assert k @ k == -one
    assert i @ j @ k == -one

def test_as_quaternion():
    # Test matrix to quaternion conversion
    Q = Matrix.from_quaternion(12, -50, 0.09, 3)
    assert Q.as_quaternion() == [12, -50, 0.09, 3]

def test_quaternion_conversion_errors():
    # Tests that program catches non-4x4 matrices
    with pytest.raises(InvalidMatrixFormat):
        A_3x2.as_quaternion()
    with pytest.raises(InvalidMatrixFormat):
        A_3x3.as_quaternion()

    # Tests that program catches matrices that were
    # not constructed by Matrix.from_quaternion()
    with pytest.raises(InvalidMatrixFormat):
        A_4x4.as_quaternion()


def test_ewise():
    ans = Matrix([[0,0,0],
                  [2,0,4],
                  [0,0,0]])
    assert ewise(A_3x3, Sing_3x3, lambda a, b: a % b) == ans

def test_MismatchedMatrixSize():
    with pytest.raises(MismatchedMatrixSize):
        A_3x2 + A_4x4

    from math import atan2
    with pytest.raises(MismatchedMatrixSize):
        ewise(A_3x3, A_4x4, atan2)


def test_list_dot_product():
    l1 = [1,.7,-13,4]
    l2 = [6,-3,0.9,7]

    ans = 6 - 2.1 - 13*.9 + 28

    v1 = Matrix([l1])
    v2 = Matrix([l2])

    assert list_dot_product(l1,l2) == sum(v1 * v2) == ans

def test_derivative_matrix():
    assert derivative_matrix(0) == Matrix([[0]])

    ans = Matrix([[0, 1, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [0, 0, 0, 3, 0, 0],
                  [0, 0, 0, 0, 4, 0],
                  [0, 0, 0, 0, 0, 5],
                  [0, 0, 0, 0, 0, 0]])
    assert derivative_matrix(5) == ans

    D = derivative_matrix(2)
    x_squared = Matrix([[99,0,1]]).T
    x = Matrix([[0,1,0]]).T
    assert D @ x_squared == 2*x


def test_determinant():
    # Test that det() raises a MatrixError for non-square matrices
    with pytest.raises(MatrixError):
        det(A_3x2)

    # Test that the determinant of A_4x4 is 0.
    assert det(A_4x4) == 0

    # Test that an invertible matrix has the correct determinant.
    assert det(Cons_4x4) == -7842

    # Test that the identity matrix has determinant 1.
    assert det(I(6)) == 1

    # Test easy 2x2
    M = Matrix([[1, 2],
                [3, 4]])
    assert det(M) == -2

    # Test base case
    K = Matrix([[77]])
    assert det(K) == 77

def test_inverse():
    # Make sure these two matrices with zero determinants raise MatrixErrors
    S = Sing_4x4.under(Fraction)
    with pytest.raises(MatrixError):
        inv(S)

    # Test that inv() raises a MatrixError for non-square matrices
    with pytest.raises(MatrixError):
        inv(A_3x2)

    # Test that that inv() produces the correct result for an invertible matrix
    C = Cons_4x4.under(Fraction)
    C_inv = inv(C)

    ans = Matrix([['-164/3921','103/3921','-563/3921', '166/3921'],
                  [ '-62/1307', '23/1307',  '90/1307',  '-1/1307'],
                  [  '91/2614','-97/2614', '208/1307', '201/1307'],
                  [  '55/1307', '85/1307','-122/1307','-231/1307']])
    ans = ans.under(Fraction)

    assert C_inv == ans

    # Test that A times its inverse is the identity matrix
    assert C @ C_inv == I(4)

def test_ref():
    assert ref(A_1x1) == Matrix([[1]])
    assert ref(Matrix([[0]])) == Matrix([[0]])
    assert ref(Matrix([[9,0,0,0]])) == Matrix([[1,0,0,0]])
    assert ref(A_3x3) == Matrix([[1,2,3],
                                 [0,1,2],
                                 [0,0,0]])
    M = Z(4,4)
    assert ref(M) == M

    M[2] = Matrix([[1,7,8,9]])
    assert ref(M) == Matrix([[1,7,8,9],
                             [0,0,0,0],
                             [0,0,0,0],
                             [0,0,0,0]])

    Tricky = Z(4,4)
    Tricky[-1,-1] = 9
    ans = Z(4,4)
    ans[0,-1] = 1
    assert ref(Tricky) == ans

    for _ in range(30):
        n = randint(1,5)
        R = realmat(n, n)
        T = intmat(n, n)
        # Check that det(A) and det(ref(A)) are either both non-zero or both zero.
        assert bool(det(ref(R))) == bool(det(R))
        assert bool(det(ref(T))) == bool(det(T))

def test_rref():
    assert rref(A_1x1) == Matrix([[1]])
    assert rref(Matrix([[0]])) == Matrix([[0]])
    assert rref(Matrix([[9,0,0,0]])) == Matrix([[1,0,0,0]])
    assert rref(A_3x3).under(Fraction) == Matrix([[1,0,-1],
                                                  [0,1, 2],
                                                  [0,0, 0]])

    assert rref(A_4x4).under(Fraction) == Matrix([[1,0,-1,-2],
                                                  [0,1, 2, 3],
                                                  [0,0, 0, 0],
                                                  [0,0, 0, 0]])

    assert round(rref(Cons_4x4), 12) == round(I(4), 12)

    # Check that rref(M) == I(M.n) <=> det(M) != 0
    for _ in range(20):
        n = randint(2,5)
        R = realmat(n,n)
        while det(R) == 0:
            R = realmat(n,n)
        assert round(rref(R), 12) == round(I(n), 12)

def test_row_reduction_and_cofactor_expansion_inverse():
    for _ in range(20):
        n = randint(2,5)
        R = realmat(n, n).under(Fraction)
        T = intmat(n, n).under(Fraction)
        # Check that det(A) and det(ref(A)) are either both non-zero or both zero.
        if det(R) != 0:
            assert round(row_reduction_inverse(R), 10) == \
                   round(cofactor_expansion_inverse(R), 10)
        if det(T) != 0:
            assert round(row_reduction_inverse(T), 10) == \
                   round(cofactor_expansion_inverse(T), 10)

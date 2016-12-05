import unittest
from linyalg.matrix import *
import fractions

__author__ = "Gideon Buckwalter"


def square_4x4(choice='A'):
    if choice == 'A':
        M = Matrix([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])
    elif choice == 'B':
        M = Matrix([[16, 15, 14, 13],
                    [12, 11, 10, 9],
                    [8, 7, 6, 5],
                    [4, 3, 2, 1]])

    elif choice == 'C':
        M = Matrix([[0,-9, 8, 7 ],
                    [12,13,14,15],
                    [-3, 5, 2, 1],
                    [6, 0, 6, 1 ]])
    else:
        raise Exception('Invalid arguments passed to function square_4x4')
    return M


def rect_3x2(choice='A'):
    if choice == 'A':
        M = Matrix([[1, 2],
                    [3, 4],
                    [5, 6]])
    elif choice == 'B':
        M = Matrix([[7, 8],
                    [9, 10],
                    [11, 12]])
    else:
        raise Exception('Invalid arguments passed to function rect_3x2')
    return M


class MatrixTests(unittest.TestCase):
    """
    Class that goes through and (hypothetically) tests each of
    matrix.Matrix's features.
    """

    def runTest(result=None):
        unittest.main()

    # Matrix class tests

    def testMatCreation(self):
        A = square_4x4()

        answer = [[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]]
        self.assertEqual(A.mat, answer)

    def testMatPrinting(self):
        M = Matrix([[134,-2,531],
                    [4,5,21526],
                    [-7,8,9]])

        ideal_str = '\n'.join(["[134, -2,   531]",
                               "[  4,  5, 21526]",
                               "[ -7,  8,     9]",
                               "3x3"])

        self.assertEqual(str(M), ideal_str)

    def testEqualityCheck(self):
        A = square_4x4()
        A_prime = square_4x4()

        # Show that two are different objects
        self.assertIsNot(A, A_prime)

        # Directly test == operator
        self.assertTrue(A == A_prime)

        # Make sure unittest.TestCase.assertEqual behaves as expected
        self.assertEqual(A, A_prime)

    def testTranspose(self):
        A = square_4x4()
        answer = Matrix([[1, 5, 9, 13],
                         [2, 6, 10, 14],
                         [3, 7, 11, 15],
                         [4, 8, 12, 16]])
        self.assertEqual(A.T, answer)

        B = rect_3x2()
        answer = Matrix([[1, 3, 5],
                         [2, 4, 6]])

        self.assertEqual(B.T, answer)

    def testSubmatrix(self):
        A = square_4x4()

        sub = A.submatrix((1, 1), (3, 3))

        answer = Matrix([[6, 7],
                         [10, 11]])

        self.assertEqual(sub, answer)

    def testAddition(self):
        A = square_4x4('A')
        B = square_4x4('B')

        C = A + B
        answer = Matrix([[17, 17, 17, 17],
                         [17, 17, 17, 17],
                         [17, 17, 17, 17],
                         [17, 17, 17, 17]])

        self.assertEqual(C, answer)

    def testSubtraction(self):
        A = square_4x4('A')
        B = square_4x4('B')

        C = A - B
        answer = Matrix([[-15, -13, -11, -9],
                         [-7, -5, -3, -1],
                         [1, 3, 5, 7],
                         [9, 11, 13, 15]])

        self.assertEqual(C, answer)

    def testElementWiseMult(self):
        A = square_4x4('A')
        B = square_4x4('B')

        C = A * B
        answer = Matrix([[16, 30, 42, 52],
                         [60, 66, 70, 72],
                         [72, 70, 66, 60],
                         [52, 42, 30, 16]])

        self.assertEqual(C, answer)

    def testMatrixMult(self):
        A = square_4x4('A')
        B = square_4x4('B')

        result = A @ B
        answer = Matrix([[80, 70, 60, 50],
                         [240, 214, 188, 162],
                         [400, 358, 316, 274],
                         [560, 502, 444, 386]])

        self.assertEqual(result, answer)

        D = rect_3x2('A')
        E = rect_3x2('B')

        result = D @ E.T
        answer = Matrix([[23, 29, 35],
                         [53, 67, 81],
                         [83, 105, 127]])

        self.assertEqual(result, answer)

        # Test chained multiplications
        result = A @ B @ A.T @ B.T

        answer = Matrix([[120080, 85520, 50960, 16400],
                         [372752, 265488, 158224, 50960],
                         [625424, 445456, 265488, 85520],
                         [878096, 625424, 372752, 120080]])

        self.assertEqual(result, answer, msg="Chained multiplication does not work.")

    def testScalarMult(self):
        A = square_4x4()

        B = 5 * A
        answer = Matrix([[5, 10, 15, 20],
                         [25, 30, 35, 40],
                         [45, 50, 55, 60],
                         [65, 70, 75, 80]])

        self.assertEqual(B, answer)

    def testUnderMethod(self):
        A = square_4x4()

        B = A.under(lambda x: x + 1)
        answer = Matrix([[2, 3, 4, 5],
                         [6, 7, 8, 9],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]])

        self.assertEqual(B, answer)

    def testMatrixNegation(self):
        A = square_4x4()

        B = -A
        answer = Matrix([[-1, -2, -3, -4],
                         [-5, -6, -7, -8],
                         [-9, -10, -11, -12],
                         [-13, -14, -15, -16]])

        self.assertEqual(B, answer)

    def testPosOperator(self):
        A = square_4x4()
        B = +(-A)

        self.assertEqual(B, A)

    def testBracketIndexing(self):
        A = square_4x4()

        row_2 = A[2]
        answer = [9, 10, 11, 12]

        self.assertEqual(row_2, answer)

        ele_2_3 = A[2][3]
        answer = 12

        self.assertEqual(ele_2_3, answer)

    def testBracketAssignment(self):
        A = square_4x4()

        A[2][3] = 999
        answer = 999

        self.assertEqual(A[2][3], answer)

    # Tests of matrix module functions

    def testEwiseFunction(self):
        A = square_4x4('A')
        B = square_4x4('B')

        C = ewise(A, B, lambda a, b: a % b)

        answer = Matrix([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [1, 3, 5, 2],
                         [1, 2, 1, 0]])

        self.assertEqual(C, answer)

    def testDotProduct(self):
        L1 = [1, 2, 3, 4]
        L2 = [-4, 3, -2, 1]

        L3 = dot_product(L1, L2)

        answer = sum([-4, 6, -6, 4])

        self.assertEqual(L3, answer)

    def testIdentityFunction(self):
        A = I(4)

        answer = Matrix([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        self.assertEqual(A, answer)

    def testMismatchedMatrixSize(self):
        with self.assertRaises(MismatchedMatrixSize):
            Matrix([[1, 2], [3, 4]]) + Matrix([[5, 6]])

    def test__rmul__OnTwoMatrices(self):
        A = square_4x4('A')
        B = square_4x4('B')

        with self.assertRaises(TypeError):
            A.__rmul__(B)

    def testExponentiation(self):
        "Tests out the __pow__ method."
        C = square_4x4('C')

        self.assertEqual(C ** 3, C @ C @ C)
        self.assertEqual(C ** -4, inv(C) @ inv(C) @ inv(C) @ inv(C))
        self.assertEqual(C ** 0, I(C.m))


    def testQuaternionInitialization(self):
        one = Matrix.from_quaternion(1, 0, 0, 0)
        i = Matrix.from_quaternion(0, 1, 0, 0)
        j = Matrix.from_quaternion(0, 0, 1, 0)
        k = Matrix.from_quaternion(0, 0, 0, 1)

        # Test quaternion to matrix conversion
        self.assertEqual(one, I(4))

        # Test quaternion multiplication properties
        self.assertEqual(i @ i, -I(4))
        self.assertEqual(j @ j, -I(4))
        self.assertEqual(k @ k, -I(4))
        self.assertEqual(i @ j @ k, -I(4))

        # Test matrix to quaternion conversion
        Q = Matrix.from_quaternion(12, -50, 0.09, 3)
        self.assertEqual(Q.as_quaternion(), [12, -50, 0.09, 3])

    def testQuaternionConversionErrors(self):
        # Tests that program catches non-4x4 matrices
        with self.assertRaises(InvalidMatrixFormat):
            rect_3x2().as_quaternion()

        # Tests that program catches matrices that were
        # not constructed by Matrix.from_quaternion()
        with self.assertRaises(InvalidMatrixFormat):
            square_4x4().as_quaternion()

    def testCheckerboardFunction(self):
        A = square_4x4('A')
        ans = A * Matrix([[1,-1,1,-1],
                          [-1,1,-1,1],
                          [1,-1,1,-1],
                          [-1,1,-1,1]])
        self.assertEqual(checkerboard(A), ans)

    def testMinorMethod(self):
        A = square_4x4('A')
        ans = Matrix([[ 1, 2, 4],
                      [ 9,10,12],
                      [13,14,16]])
        self.assertEqual(A.minor(1, 2), ans)

    def testDeterminant(self):

        # Test that det() raises a MatrixError for non-square matrices
        X = rect_3x2()
        with self.assertRaises(MatrixError):
            det(X)

        # Test that the determinant of A is 0.
        A = square_4x4()
        self.assertEqual(det(A), 0)

        # Test that an invertible matrix has the correct determinant.
        C = square_4x4('C')
        self.assertEqual(det(C), -7842)

        # Test that the identity matrix has determinant 1.
        self.assertEqual(det(I(6)), 1)

        # Test easy 2x2
        M = Matrix([[1, 2],
                    [3, 4]])
        self.assertEqual(det(M), -2)

        # Test base case
        K = Matrix([[77]])
        self.assertEqual(det(K), 77)

    def testInverse(self):

        # Make sure these two matrices with zero determinants raise MatrixErrors
        A = square_4x4('B').under(fractions.Fraction)
        with self.assertRaises(MatrixError):
            inv(A)

        B = square_4x4('B').under(fractions.Fraction)
        with self.assertRaises(MatrixError):
            inv(B)

        # Test that inv() raises a MatrixError for non-square matrices
        X = rect_3x2()
        with self.assertRaises(MatrixError):
            inv(X)

        # Test that that inv() produces the correct result for an
        # invertible matrix
        C = square_4x4('C').under(fractions.Fraction)
        C_inv = inv(C)

        ans = Matrix([['-164/3921', '103/3921', '-563/3921', '166/3921'],
                      ['-62/1307', '23/1307', '90/1307', '-1/1307'],
                      ['91/2614', '-97/2614', '208/1307', '201/1307'],
                      ['55/1307', '85/1307', '-122/1307', '-231/1307']])
        ans = ans.under(fractions.Fraction)

        self.assertEqual(C_inv, ans)

        # Test that A times its inverse is the identity matrix
        self.assertEqual(C @ C_inv, I(4))



if __name__ == '__main__':
    unittest.main()





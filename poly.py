from matrix import *
import random
from itertools import dropwhile, chain, accumulate
from functools import reduce
import math
import cmath

__author__ = "Gideon Buckwalter"

"""
Author: Gideon Buckwalter
Email: gideon.buckwalter@gmail.com
GitHub: @GideonBuckwalter
Creation Date: Dec 1 2016
Version: Python 3.5.2
"""

class Polynomial:

	def __init__(self, coefs=None, var_name="x"):
		self.coefs = coefs # ASCENDING ORDER: [1,2,3,4] -> 1 + 2x + 3x^2 + 4x^3
		self.var_name = var_name

		self.degree = len(coefs) - 1
		self.trim()

	@classmethod
	def variable(var_name="x"):
		return Polynomial(coefs=[0,1], var_name=var_name)

	def __repr__(self):
		return "Polynomial({})".format(str(self.coefs))

	def __str__(self):
		def x_to_the(n):
			if n == 1:
				return "x"
			elif n != 0:
				return "x^" + str(n)
			else:
				return ""

		def join_coef_and_var(coef, n):
			if coef == 1 and n != 0:
				return x_to_the(n)
			else:
				return str(coef) + x_to_the(n)


		return " + ".join(join_coef_and_var(coef, n)
					for n, coef in enumerate(self.coefs) if coef != 0)

	def as_vector(self, dimension=None):
		if dimension is None:
			dimension = self.degree
		if dimension < self.degree:
			raise MatrixError("Cannot clip polynomial vector to lower dimension than (degree+1).")

		extra_dims = dimension - (self.degree + 1)
		return Matrix.from_list_vector(self.coefs + [0]*extra_dims)

	def nonzero_terms(self):
		return filter(lambda coef: coef != 0, self.coefs)



	def __add__(self, other):
		if isinstance(other, Polynomial):
			degree_plus_1 = max(self.degree, other.degree) + 1
			return Polynomial((
							self.as_vector(degree_plus_1) + other.as_vector(degree_plus_1)
							).as_list_vector())
		else:
			return Polynomial([self.coefs[0] + other] + self.coefs[1:])

	def __radd__(self, other):
		return self.__add__(other)

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + -other



	def decompose(self):
		"""
		Decomposes self into a list of self's individual Polynomial terms.
		Ex: -5 + 3x + x^2  ->  [-5, 3x, x^2]
		"""
		return [Polynomial([0]*degree + [coef]) for degree, coef in enumerate(self.coefs)]

	def convolve(self, other):
		if self.coefs == [0] or other.coefs == [0]:
			return Polynomial([0])
		return sum(single_term_poly * other for single_term_poly in self.decompose())

	def __mul__(self, other):
		"""
		Super-method that figures out how to handle expresions of the form "self * other"
		where other could be either another polynomial (in which case: convolve self and other),
		or a scalar (in which case: multiply all the coefficients of self by other).
		"""
		if isinstance(other, Polynomial):
			if len(self.nonzero_terms()) == 1:
				return self.coefs[-1] * Polynomial([0]*self.degree + other.coefs)
			elif len(other.nonzero_terms()) == 1:
				return other.__mul__(self)
			else:
				return self.convolve(other)
		else:
			return Polynomial([other * coef for coef in self.coefs])

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		return 1/other * self

	def __pow__(self, exp):
		#print("__pow__(", self, ",", exp, ")")
		if exp != float(exp):
			raise TypeError("Polynomials cannot currently be raised to fractional powers.")
		if exp < 0:
			raise TypeError("Polynomials cannot currently be raised to negative powers.")
		if exp == 0:
			return Polynomial([1])
		return reduce(lambda p, q: p * q, (self for _ in range(exp)))




	def __call__(self, x):
		return sum(coef * x**n for n, coef in enumerate(self.coefs))

	def trimmed(self):
		coefs = list(reversed(list(
					dropwhile(lambda e: e == 0, reversed(self.coefs))
				)))
		return Polynomial([0] if coefs == [] else coefs)

	def trim(self):
		self.coefs = list(reversed(list(
			dropwhile(lambda e: e == 0, reversed(self.coefs))
			)))
		self.coefs = [0] if self.coefs == [] else self.coefs

	def is_constant(self):
		return len(self.coefs) == 1





def synthetic_factor_out(poly, root):
	quotient_coefs = list(reversed(list(
		   				accumulate(
		   					reversed(poly.coefs),
							lambda prev, new: root*prev + new))))

	remainder = quotient_coefs[0]
	quotient_coefs = quotient_coefs[1:]

	if abs(remainder) > 1e-9:
		raise Exception("Factoring {} from {} left a remainder of {}!".format(root, poly, remainder))

	return Polynomial(quotient_coefs)




def complex_round(z, ndigits=1):
	z = round(z.real, ndigits) + round(z.imag, ndigits)*1j
	return z.real if z.imag == 0 else z


def d_dx(poly):
	"""
	Returns the derivative of the polynomial.
	Syntax: p_prime = d_dx(p) 
	"""
	return Polynomial((
		derivative_matrix(poly.degree) @ poly.as_vector()
			).as_list_vector())




def newtons_method(poly, x0=0, tolerance="Coefficient Average / 1e12", max_iters=1000):

	iterations = 0

	# Set up default value for tolerance
	if tolerance == "Coefficient Average / 1e12":
		t = sum(map(abs, poly.coefs))/len(filter(lambda e: e != 0, poly.coefs)) * 1e-12
		tolerance = t if t != 0 else 1e-12

	x = x0
	x_prev = x - 1 # just give it a value so that the initial error isn't 0.

	error = abs(x - x_prev)

	f = poly
	f_prime = d_dx(poly)

	while error > tolerance:
		if iterations > max_iters:
			raise NumericalSolverError("Newton's method has not found a zero; max iterations reached.")

		iterations += 1
		
		try:
			x_next = x - f(x)/f_prime(x)
			#print("\tx0 =", x0, "\tx =", x, "\tp(x) =", poly(x))
		except ZeroDivisionError:
			fuzz = 1e-12
			x_next = x - f(x + fuzz)/f_prime(x + fuzz)

		error = abs(x - x_prev)
		x_prev, x = x, x_next
	#print("Newton's method iterations:", iterations)
	return x



def real_zero_bounds(poly):
	"""Source: http://faculty.uncfsu.edu/fnani/FicamsFrontpage/ch3.6.pdf"""

	poly = poly / poly.coefs[-1] # Normalize leading coefficient
	constant_term = poly.coefs[0]

	Bound = min(
				max(1, sum(map(abs, poly.coefs))),
				1 + max(map(abs, poly.coefs))
			)

	return (-Bound, Bound)






def roots(poly, correct_digits=12, max_newton_iters=100, show_multiple_roots=False):

	zeros = []
	attempts = 0

	while not poly.is_constant():
		
		bounds = real_zero_bounds(poly)
		x0 = cmath.rect( # Choose a random COMPLEX number at which to start the search.
				random.random() * bounds[1],
				random.random() * 2 * math.pi)

		#print("x0 =", x0)

		try:
			new_zero = newtons_method(poly,
									  x0=x0,
									  tolerance=10**-(correct_digits + poly.degree),
									  max_iters=max_newton_iters)
		except NumericalSolverError: # Newton's Method got stuck, try with different x0.
			attempts += 1
			if attempts <= 100:
				continue
			else:
				raise NumericalSolverError("Randomized intitial guesses have failed to reveal more roots.")

		zeros.append(complex_round(new_zero, correct_digits))
		#print("Intermediate value of zeros:", zeros)
		poly = synthetic_factor_out(poly, new_zero)
		#print("Polynomial after synthetic division:", poly)
		attempts = 0 # Reset attempts back to zero

	return zeros if show_multiple_roots else set(zeros)




class NumericalSolverError(Exception):
	def __init__(self, value):
		self.value = value





if __name__ == '__main__':

	def test_out(poly):
		print()
		print("P =", poly)
		#print("P is a constant function." if poly.is_constant() else "P is not a constant function.")
		#print("P's real zeros are bounded by", real_zero_bounds(poly))
		print("P(x) = 0 when x =", roots(poly, show_multiple_roots=False))
	
	"""
	parabola = Polynomial([0,0,1])
	#test_out(parabola)

	x = Polynomial("x")
	print((x + 1)**3)

	cubic = Polynomial([1,1,1,1])
	print(cubic)
	#test_out(cubic)

	#test1 = Polynomial([0., -5040., 13068., -13132., 6769., -1960., 322., -28., 1])
	#test_out(test1)

	A = Polynomial([-12,-20,-5,5,2])
	test_out(A)

	imag = Polynomial([1,0,1])
	test_out(imag)

	imag_and_real = Polynomial([-1,1,-1,1])
	test_out(imag_and_real)

	const = Polynomial([3,0,0,0,0,0])
	test_out(const)
	
	p = Polynomial([374.01, 4.492, -47.182, -1.624, 1]) # (x - 3) (x + 5) (x - 7.124) (x + 3.5)
	test_out(p)

	# Newton's method starts out where d/dx[q] == 0.
	q = Polynomial([1.1,0,-1]) # 1 - x^2
	test_out(q)
	
	print()

	"""







	x = Polynomial("x")

	binomial = x + 1
	print("binomial:	", binomial)
	print("binomial^2:	", binomial ** 2)
	print("binomial^5:	", binomial ** 5)

	print(repr(0 * binomial))

	quadratic = 1 + 3*x - 5*x**2
	print("The roots of", quadratic, "occur when x =", roots(quadratic))















	print("\n\n\n\n\n")
"""polynomial_arithmetic.py
A Python module to handle polynomial arithmetic and operations in the quotient ring
Z_a[x]/f(x).
"""
from math import log, floor, ceil, sqrt, gcd
import numpy as np
import random
from decimal import *

def print_and_return(variable, value):
    """Self explanatory.
    """
    print(variable, "=", value); return value

def two_power(value):
    """Returns largest power of two <= value as a readable string.
    """
    return "2^" + str(floor(log(value, 2)))

def generate_gaussian_distribution(length, mean, standard_deviation):
    """Generates a Gaussian/Normal polynomial.
    
    Generates a normally distributed polynomial in the ring Z[x]/(f(x)), where f = x^length + 1,
    with each entry being independently identically distributed as Normal(mean, standard_deviation).
    Rounds the coefficients to the nearest integer.

        Args:
            length (int): Equals degree + 1 of the polynomial. f = x^length + 1 holds to.
            mean (float): Mean of the normal distribution.
            standard_deviation (float): Standard deviation of the normal distr.

        Returns:
            A gaussian polynomial in the polynomial ring.
    """
    A = np.round(np.random.normal(mean, standard_deviation, length)).astype(np.int64).tolist()
    return Polynomial(length, A)

def generate_uniform_distribution(length, low_bound, high_bound):
    """Generates a Uniform polynomial.
    
    Generates a uniformly distributed polynomial in the ring Z[x]/(f(x)), where f = x^length + 1,
    with each coefficient being independently uniformly distributed in [lower_bound, high_bound].
    Rounds the coefficients to the nearest integer.

        Args:
            length (int): Equals degree + 1 of the polynomial. f = x^length + 1 holds to.
            low_bound (float): Minimum possible value of the entries.
            high_bound (float): Maximum possible value of the entries.

        Returns:
            A uniform polynomial in the polynomial ring.
    """
    return Polynomial(length, [random.randint(low_bound, high_bound) for i in range(length)])

def generate_ternary_distribution(length):
    """Generates a Ternary polynomial.
    
    Generates a ternaryly distributed polynomial in the ring Z[x]/(f(x)), where f = x^length + 1,
    with each coefficient being independently uniformly distributed in {-1, 0, 1}. 

        Args:
            length (int): Equals degree + 1 of the polynomial. f = x^length + 1 holds to.
            
        Returns:
            A ternary polynomial in the polynomial ring.
    """
    return Polynomial(length, [random.randint(-1, 1) for i in range(length)])

def generate_constant_poly(ring_dim, constant, coeff_modulus = None):
    """Generates a constant polynomial c in the polynomial ring.
    
    Generates a constant polynomial in the ring Z[x]/f(x), where f(x) = x^(ring_dim) + 1.
    Reduces modulus coeff_modulus = q if desired.

        Args:
            ring_dim (int): Exponent of the quotient polynomial x^(ring_dim) + 1.
            constant (int): Constant c to be treated as a polynomial
            coeff_modulus (int): Modulus q of the ring.
            
        Returns:
            c in Z_q[x]/f(x) as a polynomial-class object.
    """
    constantcoeffs = [0 for _ in range(ring_dim)]
    constantcoeffs[0] = constant
    if coeff_modulus:
        constantcoeffs[0] = constant % coeff_modulus
    return Polynomial(ring_dim, constantcoeffs)


def generate_one_poly(ring_dim):
    """Generates 1 in the polynomial ring.
    
    Generates a one polynomial in the ring Z[x]/f(x), where f(x) = x^(ring_dim) + 1.

        Args:
            ring_dim (int): Exponent of the quotient polynomial x^(ring_dim) + 1.
            
        Returns:
            1 in Z[x]/f(x) as a polynomial-class object.
    """
    onecoeffs = [0 for _ in range(ring_dim)]
    onecoeffs[0] = 1
    return Polynomial(ring_dim, onecoeffs)

def generate_zero_poly(ring_dim):
    """Generates 0 in the polynomial ring.
    
    Generates a zero polynomial in the ring Z[x]/f(x), where f(x) = x^(ring_dim) + 1.

        Args:
            ring_dim (int): Exponent of the quotient polynomial x^(ring_dim) + 1.
            
        Returns:
            0 in Z[x]/f(x) as a polynomial-class object.
    """
    return Polynomial(ring_dim, [0 for _ in range(ring_dim)])

def generate_monomial(ring_dim, exponent):
    """Generates a monomial in the polynomial ring.
    
    Generates the monomial x^exponent in the ring Z[x]/f(x), where f(x) = x^(ring_dim) + 1.

        Args:
            ring_dim (int): Exponent of the quotient polynomial x^(ring_dim) + 1.
            exponent (int): Exponent of the desired monomial.
            
        Returns:
            x^exponent in Z[x]/f(x) as a polynomial-class object.
    """
    exponent = exponent % (2*ring_dim)
    monomial_coeffs = [0 for _ in range(ring_dim)]
    if exponent >= ring_dim:
        monomial_coeffs[exponent - ring_dim] = -1
    else:
        monomial_coeffs[exponent] = 1
    return Polynomial(ring_dim, monomial_coeffs)

def rotation_poly(ring_dim, alpha, modulus):
    """Generates the rotation polynomial X^(alpha) - 1 in the polynomial ring.
    
    Generates the polynomial X^(alpha) - 1 in the modular quotient ring Z_q[x]/f(x),
    where f(x) = x^(ring_dim) + 1 and q = modulus.

        Args:
            ring_dim (int): Exponent of the quotient polynomial x^(ring_dim) + 1.
            alpha (int): Exponent of the polynomial. Can be negative!
            modulus (int): Modulus, by which the coefficients of the result get reduced.
            
        Returns:
            X^(alpha) - 1 in Z[x]/f(x) as a polynomial-class object.
    """
    a = [0 for _ in range(ring_dim)]
    quotient = alpha // ring_dim
    residue = alpha % ring_dim
    a[residue] = ((-1) ** abs(quotient)) % modulus
    a[0] = (a[0] - 1) % modulus
    return Polynomial(ring_dim, a)

def dot_product(array1, array2):
    """Computes the dot product of two arrays.
    
    Calculates <array1, array2>, the dot product, as vectors.

        Args:
            array1 (array): Array 1.
            array2 (array): Array 2.
            
        Returns:
            The sum of k=1 to n of (array1[k] * array2[k]).
    """
    return sum([array1[i] * array2[i] for i in range(0, min(len(array1),len(array2)))])

def mod(array_or_integer, modulus):
    """Computes the centered reduction of an array/integer.
    
    Computes for an integer an unique reduction representant modulo modulus, which lies in the
    half-open interval (-modulus/2, modulus/2]. For an array the same coefficient-wise.

        Args:
            array_or_integer (array/int): Array/Integer to get reduced.
            modulus (int): Modulus for reduction.
            
        Returns:
            An reduced array/integer.
    """
    a, q = array_or_integer, modulus
    if type(a) == list:
        mod = [a[i] % q for i in range(len(a))]
        for i in range(len(a)):
            if mod[i] > q/2:
                mod[i] -= q
    if type(a) == int:
        mod = a % q
        if mod > q/2:
            mod -= q
    return mod

def sign(x):
    """ Self explanatory.
    """
    if x < 0:
        return -1
    return 1
    
def multiply_poly_as_array(poly1, poly2):
    """Multiplys two arrays, which represent polynomials.
       Auxiliary function for the karatsuba algorithm.

        Args:
            poly1 (Array): Array of coefficients of polynomial one.
            poly2 (Array): Array of coefficients of polynomial two.

        Returns:
            An array of coefficients of the product of the two polynomials.
    """
    length = len(poly1)
    new_coeffs = [0 for _ in range(2*length)]
    for i in range(0, length):
        for j in range(0, length):
            new_coeffs[i+j] += poly1[i]*poly2[j] 
    return new_coeffs

def karatsuba(poly1, poly2, ring_degree, param):
    """Multiplys two polynomials with the Karatsuba algorithm.
       Auxiliary function for the karatsuba algorithm.

        Args:
            poly1 (Array): Polynomial one.
            poly2 (Array): Polynomial two.
            ring_degree(int of type 2^k): Ring degree of underlying Polynomial ring. We need to keep track of this,
                                          since arrays get smaller per call to boost the runtime.
            param(int of type 2^k): Karatsuba parameter. Starts equal as ring degree, gets divided in half
                                    every recursive call.

        Returns:
            An array of coefficients of the product of the two polynomials.
    """
    
    # Base case: As determined manually, karatsuba is only slower for powers of two smaller than 2^3.
    if param == 2**2:
        return multiply_poly_as_array(poly1, poly2)
    
    # Divide step: Splitting up the array in two half-sized ones.
    param_new = param // 2
    a_0, a_1, b_0, b_1 = poly1[:param_new], poly1[param_new:], poly2[:param_new], poly2[param_new:]
    
    # Conquer step: Recursively calling karatsuba 3 times, which yields the running time of O(n^(log_2(3))).
    c_2 = karatsuba(a_1, b_1, ring_degree, param_new)
    c_0 = karatsuba(a_0, b_0, ring_degree, param_new)
    sum1 = [(a_0[i] + a_1[i])  for i in range(param_new)]
    sum2 = [(b_0[i] + b_1[i])  for i in range(param_new)]
    c_1 = karatsuba(sum1, sum2, ring_degree, param_new)
    c_1 = [(c_1[i] - c_2[i] - c_0[i]) for i in range(param)]

    zeros_param = [0] * param
    zeros_param_new = [0] * param_new
    
    # Recombining the arrays: A special case for param = ring_degree occurs, since then X^(param) = -1.
    if ring_degree == param:
        c_2 = [(-c_2[i]) for i in range(param)]
        c_1 = [c_1[(i - param_new) % param] for i in range(param)]
        for i in range(param_new):
                c_1[i] *= -1
        return [(c_0[i] + c_1[i] + c_2[i]) for i in range(param)]     
    else:
        c_1 = zeros_param_new + c_1 + zeros_param_new # new length = 2*param
        c_2 = zeros_param + c_2
        c_0 = c_0 + zeros_param
        return [(c_0[i] + c_1[i] + c_2[i]) for i in range(2*param)]
    
class Polynomial:  
    """A polynomial in the ring R_a.

    Here, R is the quotient ring Z[x]/f(x), where f(x) = x^d + 1.
    The polynomial keeps track of the ring degree d, the coefficient
    modulus a, and the coefficients in an array.

    Attributes:
        ring_degree (int): Degree d of polynomial that determines the
            quotient ring R.
        coeffs (array): Array of coefficients of polynomial, where coeffs[i]
            is the coefficient for x^i.
    """

    def __init__(self, degree, coeffs):
        """Inits Polynomial in the ring R_a with the given coefficients.

        Args:
            degree (int): Degree of quotient polynomial for ring R_a.
            coeffs (array): Array of integers of size degree, representing
                coefficients of polynomial.
        """
        self.ring_degree = degree
        assert len(coeffs) == degree, 'Size of polynomial array %d is not \
            equal to degree %d of ring' %(len(coeffs), degree)

        self.coeffs = coeffs

    def add(self, poly, coeff_modulus=None):
        """Adds two polynomials in the ring.

        Adds the current polynomial to poly inside the ring R_a.

        Args:
            poly (Polynomial): Polynomial to be added to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial which is the sum of the two polynomials.
        """
        assert isinstance(poly, Polynomial)

        poly_sum = Polynomial(self.ring_degree, [0] * self.ring_degree)

        poly_sum.coeffs = [self.coeffs[i] + poly.coeffs[i] for i in range(self.ring_degree)]
        if coeff_modulus:
            poly_sum = poly_sum.mod(coeff_modulus)
        return poly_sum

    def subtract(self, poly, coeff_modulus=None):
        """Subtracts second polynomial from first polynomial in the ring.

        Computes self - poly.

        Args:
            poly (Polynomial): Polynomial to be added to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial which is the difference between the two polynomials.
        """
        assert isinstance(poly, Polynomial)

        poly_diff = Polynomial(self.ring_degree, [0] * self.ring_degree)

        poly_diff.coeffs = [self.coeffs[i] - poly.coeffs[i] for i in range(self.ring_degree)]
        if coeff_modulus:
            poly_diff = poly_diff.mod(coeff_modulus)
        return poly_diff

    def multiply(self, poly, coeff_modulus = None):
        """Multiplies two polynomials in the ring using the Karatsuba algorithm.

        Multiplies the current polynomial to poly inside the ring R_a
        using the Karatsuba Algorithm in O(n^(log_2(3))).

        Args:
            poly (Polynomial): Polynomial to be multiplied to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial which is the product of the two polynomials.
        """
        N = self.ring_degree
        
        if (2**floor(log(N,2)) == N) & (N >= 2**3):
            K = karatsuba(self.coeffs, poly.coeffs, N, N)
            # Reducing only after Karatsuba yields a significantly better running time.
            if coeff_modulus:
                return Polynomial(N, [K[i] % coeff_modulus for i in range(N)])
            return Polynomial(N, K)
        
        return self.multiply_naive(poly, coeff_modulus)            

    def multiply_naive(self, poly, coeff_modulus=None):
        """Multiplies two polynomials in the ring in O(n^2).

        Multiplies the current polynomial to poly inside the ring R_a
        naively in O(n^2) time.

        Args:
            poly (Polynomial): Polynomial to be multiplied to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial which is the product of the two polynomials.
        """
        assert isinstance(poly, Polynomial)

        poly_prod = Polynomial(self.ring_degree,
                               [0] * self.ring_degree)

        for d in range(2 * self.ring_degree - 1):
            # Since x^d = -1, the degree is taken mod d, and the sign
            # changes when the exponent is > d.
            index = d % self.ring_degree
            sign = int(d < self.ring_degree) * 2 - 1

            # Perform a convolution to compute the coefficient for x^d.
            coeff = 0
            for i in range(self.ring_degree):
                if 0 <= d - i < self.ring_degree:
                    coeff += self.coeffs[i] * poly.coeffs[d - i]
            poly_prod.coeffs[index] += sign * coeff
            
            if coeff_modulus:
                poly_prod.coeffs[index] %= coeff_modulus

        return poly_prod

    def scalar_multiply(self, scalar, coeff_modulus=None):
        """Multiplies polynomial by a scalar.

        Multiplies the current polynomial to scalar inside the ring R_a.

        Args:
            scalar (int): Scalar to be multiplied to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial which is the product of the polynomial and the
            scalar.
        """
        if coeff_modulus:
            new_coeffs = [(scalar * c) % coeff_modulus for c in self.coeffs]
        else:
            new_coeffs = [(scalar * c) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def scalar_integer_divide(self, scalar, coeff_modulus=None):
        """Divides polynomial by a scalar.

        Performs integer division on the current polynomial by the scalar inside
        the ring R_a.

        Args:
            scalar (int): Scalar to be divided by.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial which is the quotient of the polynomial and the
            scalar.
        """
        if coeff_modulus:
            new_coeffs = [(c // scalar) % coeff_modulus for c in self.coeffs]
        else:
            new_coeffs = [(c // scalar) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)
    
    def rounded_integer_divide(self, scalar, coeff_modulus=None):
        """Divides polynomial by a scalar and rounds to the nearest integer.

        Performs integer division on the current polynomial by the scalar inside
        the ring R_a. Rounds the coefficients afterwards. 

        Args:
            scalar (int): Scalar to be divided by.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A rounded Polynomial which is the quotient of the polynomial and the
            scalar.
        """
        
        if coeff_modulus:
            getcontext().prec = ceil(log(coeff_modulus, 10))
            new_coeffs = [round(Decimal(c) / Decimal(scalar)) % coeff_modulus for c in self.coeffs]
        else:
            
            getcontext().prec = ceil(log(self.norm()+2, 10))
            new_coeffs = [round(Decimal(c) / Decimal(scalar)) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)
        

    def variable_powering(self, r):
        """Powers X by r.

        Powers the variable X by r. We do so by applying the transformation m(X) -> m(X^r),
        which is an automorphism for r in the units of Z_(2*N).

        Returns:
            A morphed Polynomial.
        """
        
        if r == 1:
            return self
                    
        new_coeffs = [0] * self.ring_degree
        for i in range(self.ring_degree):
            index = (i * r) % (2 * self.ring_degree)
            if index < self.ring_degree:
                new_coeffs[index] += self.coeffs[i]
            else:
                new_coeffs[index - self.ring_degree] += -self.coeffs[i]
                   
        return Polynomial(self.ring_degree, new_coeffs)

    def round(self):
        """Rounds all coefficients to nearest integer.

        Rounds all the current polynomial's coefficients to the nearest
        integer, where |x| = n + 0.5 rounds to |x| = n
        (i.e. 0.5 rounds to 0 and -1.5 rounds to -1).

        Returns:
            A Polynomial which is the rounded version of the current
            polynomial.
        """
        if type(self.coeffs[0]) == complex:
            new_coeffs = [round(c.real) for c in self.coeffs]
        else:
            new_coeffs = [round(c) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)


    def mod(self, coeff_modulus):
        """Mods all coefficients in the given coefficient modulus.

        Mods all coefficients of the current polynomial using the
        given coefficient modulus.

        Args:
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial whose coefficients are modulo coeff_modulus.
        """
        new_coeffs = [c % coeff_modulus for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def mod_small(self, coeff_modulus):
        """Turns all coefficients in the given coefficient modulus
        to the range (-q/2, q/2].

        Turns all coefficients of the current polynomial
        in the given coefficient modulus to the range (-q/2, q/2].

        Args:
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            A Polynomial whose coefficients are modulo coeff_modulus.
        """
        try:
            new_coeffs = [c % coeff_modulus for c in self.coeffs]
            new_coeffs = [c - coeff_modulus if c > coeff_modulus // 2 else c for c in new_coeffs]
        except:
            print(self.coeffs)
            print(coeff_modulus)
            new_coeffs = [c % coeff_modulus for c in self.coeffs]
            new_coeffs = [c - coeff_modulus if c > coeff_modulus // 2 else c for c in new_coeffs]
        return Polynomial(self.ring_degree, new_coeffs)
    
    def base_decompose(self, base, num_levels):
        """Decomposes each polynomial coefficient into a base B
        representation.

        Args:
            base (int): Base to decompose coefficients with.
            num_levels (int): Log of ciphertext modulus with base B.

        Returns:
            An array of Polynomials, where the ith element is the coefficient of
            the base B^i.
        """
        N = self.ring_degree
        decomposed = [Polynomial(N, [0] * N) for _ in range(num_levels)]
        poly = self

        for i in range(num_levels):
            decomposed[i] = poly.mod(base)
            poly_coeffs = getattr(poly, 'coeffs')
            poly = Polynomial(N, [poly_coeffs[i] // base for i in range(N)])
            
        return decomposed    
    
    def is_equal_to(self, poly, coeff_modulus=None):
        """Compares to polynomials in the ring.

        Compares coefficient-wise polynomials in the ring R_a.

        Args:
            poly (Polynomial): Polynomial to be compared to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            True, if they are (essentially i.e. modular-wise) the same.
            False, if they are not the same.
        """
        assert self.ring_degree == poly.ring_degree
        
        for i in range(self.ring_degree):
            if coeff_modulus:
                if (self.coeffs[i] % coeff_modulus) != (poly.coeffs[i] % coeff_modulus):
                    return False
            elif self.coeffs[i] != poly.coeffs[i]:
                return False
        return True
    
    def norm(self, coeff_modulus = None):
        """Computes the maximum norm.

        Computes the maximum norm of a polynomial after reducing in the ring R_a.

        Args:
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.

        Returns:
            The absolute value of the largest coefficient (after modular reducing).
        """
        if coeff_modulus:
            new_coeffs = [self.coeffs[i] % coeff_modulus for i in range(self.ring_degree)]
            self = Polynomial(self.ring_degree, new_coeffs)
        norm = 0
        for i in range(self.ring_degree):
            if abs(self.coeffs[i]) > norm:
                norm = abs(self.coeffs[i])
        return round(norm)
    
    def evaluate(self, inp):
        """Evaluates the polynomial at the given input value.
        Evaluates the polynomial using Horner's method.
        Args:
            inp (int): Value to evaluate polynomial at.
        Returns:
            Evaluation of polynomial at input.
        """
        result = self.coeffs[-1]

        for i in range(self.ring_degree - 2, -1, -1):
            result = result * inp + self.coeffs[i]

        return result

    def __str__(self):
        """Represents polynomial as a readable string.

        Returns:
            A string which represents the Polynomial.
        """
        s = ''
        for i in range(self.ring_degree - 1, -1, -1):
            if self.coeffs[i] != 0:
                if s != '':
                    s += ' + '
                if i == 0 or self.coeffs[i] != 1:
                    s += str(int(self.coeffs[i]))
                if i != 0:
                    s += 'x'
                if i > 1:
                    s += '^' + str(i)
        return s
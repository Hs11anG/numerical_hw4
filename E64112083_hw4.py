import numpy as np
import scipy.integrate as integrate
from numpy.polynomial.legendre import leggauss

def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# Define integration boundaries
def lower_bound(x):
    return np.sin(x)

def upper_bound(x):
    return np.cos(x)

def exact_integral():
    def inner_integral(x):

        term1 = (upper_bound(x)**2 - lower_bound(x)**2) * np.sin(x) 
        term2 = (upper_bound(x) - lower_bound(x)) * np.cos(x)**2     
        return term1 + term2
    
    
    result, error = integrate.quad(inner_integral, 0, np.pi/4)
    return result

# a. Using Simpson's rule (n=4, m=4)
def simpson_double_integral(n, m):
    # Simpson's rule requires n to be even, here we use n=4
    # Partition for outer integral
    x_points = np.linspace(0, np.pi/4, n+1)
    dx = x_points[1] - x_points[0]
    
    result = 0
    for i in range(n+1):
        x = x_points[i]

        y_lower = lower_bound(x)
        y_upper = upper_bound(x)
        y_points = np.linspace(y_lower, y_upper, m+1)
        dy = (y_upper - y_lower) / m

        inner_weights = np.ones(m+1)
        inner_weights[1:m:2] = 4  
        inner_weights[2:m:2] = 2 
        
        inner_sum = sum(inner_weights[j] * f(x, y_points[j]) for j in range(m+1))
        inner_integral = inner_sum * dy / 3

        if i == 0 or i == n:
            weight = 1
        elif i % 2 == 1:  # odd points
            weight = 4
        else:  # even points
            weight = 2
        
        result += weight * inner_integral
    
    return result * dx / 3

# b. Using Gaussian Quadrature (n=3, m=3)
def gaussian_quadrature_double_integral(n, m):
   
    x_points, x_weights = leggauss(n)
    
    a, b = 0, np.pi/4
    x_points = 0.5 * (b - a) * x_points + 0.5 * (b + a)
    x_weights = 0.5 * (b - a) * x_weights
    
    result = 0
    for i in range(n):
        x = x_points[i]

        y_lower = lower_bound(x)
        y_upper = upper_bound(x)

        y_points, y_weights = leggauss(m)

        y_points = 0.5 * (y_upper - y_lower) * y_points + 0.5 * (y_upper + y_lower)
        y_weights = 0.5 * (y_upper - y_lower) * y_weights
        
       
        inner_sum = sum(y_weights[j] * f(x, y_points[j]) for j in range(m))
        
        result += x_weights[i] * inner_sum
    
    return result

# Calculate results
exact_value = exact_integral()
simpson_result = simpson_double_integral(4, 4)
gaussian_result = gaussian_quadrature_double_integral(3, 3)

print("Problem 3 Results:")
print(f"a. Simpson's rule (n=4, m=4): {simpson_result:.8f}")
print(f"b. Gaussian Quadrature (n=3, m=3): {gaussian_result:.8f}")
print(f"c. Exact value: {exact_value:.8f}")


# Problem 4: Using composite Simpson's rule to calculate two improper integrals
def transform_integral_a(n):

    h = 1.0 / n
    x_points = np.linspace(0, 1, n+1)

    start_point = h/1000
    
    integrand = lambda x: x**(-1/4) * np.sin(x)

    reference, _ = integrate.quad(integrand, start_point, 1)
    
    # Composite Simpson's rule
    weights = np.ones(n+1)
    weights[1:n:2] = 4  
    weights[2:n:2] = 2  

    function_values = np.zeros(n+1)
    function_values[0] = integrand(start_point)
    for i in range(1, n+1):
        function_values[i] = integrand(x_points[i])
    
    simpson_result = h/3 * np.sum(weights * function_values)
    
    return simpson_result, reference

def transform_integral_b(n):

    
    h = 1.0 / n
    t_points = np.linspace(0, 1, n+1)

    start_point = h/1000
    
    transformed_integrand = lambda t: t**2 * np.sin(1/t)

    original_integrand = lambda x: x**(-4) * np.sin(x)
    reference, _ = integrate.quad(original_integrand, 1, np.inf)
    
    # Composite Simpson's rule
    weights = np.ones(n+1)
    weights[1:n:2] = 4 
    weights[2:n:2] = 2 

    function_values = np.zeros(n+1)
    function_values[0] = transformed_integrand(start_point)
    for i in range(1, n+1):
        function_values[i] = transformed_integrand(t_points[i])
    
    simpson_result = h/3 * np.sum(weights * function_values)
    
    return simpson_result, reference

# Calculate Problem 4 results
n = 4
result_a, ref_a = transform_integral_a(n)
result_b, ref_b = transform_integral_b(n)

print("Problem 4 Results:")
print(f"a. ∫₀¹ x^(-1/4) sin(x) dx using composite Simpson's rule (n=4): {result_a:.8f}")
print(f"b. ∫₁^∞ x^(-4) sin(x) dx using transformed composite Simpson's rule (n=4): {result_b:.8f}")

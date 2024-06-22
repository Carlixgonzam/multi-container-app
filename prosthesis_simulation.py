# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:07:41 2024

@author: carla
"""

from fenics import *
import matplotlib.pyplot as plt


# Crear malla y definir función de espacio
mesh = UnitCubeMesh(10, 10, 10)  # Aquí deberías definir la geometría específica de tu prótesis
V = VectorFunctionSpace(mesh, 'P', 1)

# Definir las condiciones de contorno
def clamped_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0)

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Definir las propiedades del material
E = 10e9  # Módulo de Young (Pa)
nu = 0.3  # Coeficiente de Poisson
mu = E / (2.0 * (1.0 + nu))
lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Definir la función de desplazamiento
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
I = Identity(d)  # Identidad tensorial
F = I + grad(u)  # Deformación gradiente
C = F.T * F  # Tensor de Cauchy-Green
E = 0.5 * (C - I)  # Tensor de Green-Lagrange
S = lambda_ * tr(E) * I + 2.0 * mu * E  # Tensor de tensiones de segundo Piola-Kirchhoff
P = F * S  # Tensor de tensiones de Piola-Kirchhoff
F_int = inner(P, grad(v)) * dx  # Fuerza interna

# Definir la carga externa (ejemplo: una fuerza constante en la dirección y)
b = Constant((0, 0, -1e6))  # Fuerza externa (N)
F_ext = dot(b, v) * dx  # Fuerza externa

# Resolver el problema variacional
u = Function(V)
solve(lhs(F_int) == rhs(F_ext), u, bc)

# Plotear el resultado
plt.figure()
plot(u, title='Displacement')
plt.show()

# Guardar el resultado en un archivo
file = File("displacement.pvd")
file << u

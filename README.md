# A finite difference approach to solving the Navier-Stokes equations for the 2-D Lid Driven Cavity problem

# Introduction
The Navier-Stokes equations are a set of non linear, partial differential equations
that govern the viscous motion of fluids. If used correctly, these equations can be used as a
powerful tool to model the flow of fluids in many different scenarios, including weather
systems, viscous fluids moving around barriers and even in computer game simulations. Due
to the non linear nature of these equations there are very few analytically solvable
problems that have been found. Due to this, these equations must be approximately solved
by numerical methods and their solutions calculated with the help of a computer. I will be
using MATLAB to implement and execute my numerical method for solving these equations
for my cavity problem.

# The Problem
The Lid Driven Cavity problem has been thoroughly studied and results from
different numerical methods approaches agree and are well documented. The problem
involves a cavity (usually square or rectangular) where there are non slip (zero velocity)
boundary conditions on three edges and a constant velocity imposed on the fourth. Figure 1
shows a diagrammatic view of this. This problem is commonly used to test out the accuracy
of newly written fluid mechanics software due to its simple boundary and initial conditions. I
will be modelling the time evolution of the velocity field for 10 seconds in intervals of 0.01
seconds and superimposing it onto a 15x15 cavity by use of a quiver plot. I will also plot the
vorticity of the flow using a contour plot and will produce a separate surface plot of the
pressure field for the cavity.


# Please find the full report in the repository above

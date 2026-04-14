# 1D Unsteady Pipe Flow Analysis

This project provides a numerical solution for 1D unsteady fluid flow in a pipe using the Navier-Stokes equations. It compares three different finite difference schemes to evaluate their stability and accuracy.

## Numerical Methods Implemented
* **FTCS** (Forward Time Central Space) - Explicit Scheme
* **BTCS** (Backward Time Central Space) - Implicit Scheme, solved using the Thomas Algorithm (TDMA)
* **Crank-Nicolson** - Semi-implicit Scheme, solved using TDMA

## Key Features
* Computes dimensionless velocity profiles over time.
* Calculates theoretical and numerical flow rates and wall shear stresses.
* Resolves the coordinate singularity at the pipe center using L'Hôpital's Rule.

## How to Run
1. Ensure you have the required libraries: `pip install numpy matplotlib`
2. Run the analysis: `1D_Unsteady_Pipe.py`

# Symmetric Tensor Decomposition using Gradient Descent

This repository provides a Python implementation of a gradient descent algorithm for symmetric tensor decomposition, as described in the paper ["Gradient Descent for Symmetric Tensor Decomposition"](https://doi.org/10.4208/aam.OA-2021-0090) by Jian-Feng Cai, Haixia Liu, and Yang Wang, published in *Annals of Applied Mathematics*.

## Overview

Tensor decomposition is a powerful tool for analyzing multi-dimensional data, extending concepts from matrix decomposition to higher-order tensors. This repository focuses on the decomposition of symmetric tensors, specifically targeting the best rank-one approximation using a gradient descent algorithm. The algorithm is designed to converge to the global minimum with high probability, particularly for orthogonally decomposable tensors.

## Features

- **Symmetric Tensor Decomposition**: Implements the CP (Canonical Polyadic) decomposition for symmetric tensors.
- **Gradient Descent Optimization**: The algorithm applies gradient descent with a carefully designed initialization, ensuring convergence to the global minimizer for the best rank-one approximation.
- **Geometric Landscape Analysis**: Explores the geometric landscape of the nonconvex optimization, showing that local minima correspond to factors in the tensor decomposition.
- **Greedy Approach**: After each rank-one approximation, the residual tensor is updated, allowing the extraction of additional components iteratively.

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/symmetric-tensor-decomposition.git
cd symmetric-tensor-decomposition

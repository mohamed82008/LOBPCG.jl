# LOBPCG

[![Build Status](https://travis-ci.org/mohamed82008/LOBPCG.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/LOBPCG.jl)

[![Coverage Status](https://coveralls.io/repos/mohamed82008/LOBPCG.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mohamed82008/LOBPCG.jl?branch=master)

[![codecov.io](http://codecov.io/github/mohamed82008/LOBPCG.jl/coverage.svg?branch=master)](http://codecov.io/github/mohamed82008/LOBPCG.jl?branch=master)

## Introduction

This package implements the algorithm for finding minimum or maximum generalized eigenvalues and vectors from "Andrew V. Knyazev. Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate
Gradient Method. SIAM Journal on Scientific Computing, 23(2):517-541, 2001." The algorithm tries to maximize or minimize the Rayleigh quotient using an enhanced conjugate gradient method.

## Example

```julia

julia> using LOBPCG

julia> using BenchmarkTools

julia> max_error_norm(A, B, x, lambda) = sqrt(maximum(sum((x)->(conj(x)*x), A*x-B*x*diagm(lambda), 1)))
max_error_norm (generic function with 1 method)

julia> max_error_norm(A, x, lambda) = sqrt(maximum(sum((x)->(conj(x)*x), A*x-x*diagm(lambda), 1)))
max_error_norm (generic function with 2 methods)

julia> T = Float64
Float64

julia> n = 10000
10000

julia> A = 10*sprand(T, n, n, 10/n);

julia> A = A' + A + 500I;

julia> B = 10*sprand(T, n, n, 10/n);

julia> B = B' + B + 500I;

julia> tol = T(1e-4)
0.0001

julia> X0 = zeros(T, n, 2);

julia> lambda, x = lobpcg(A, B, X0, false, tol=tol, maxiter=Inf);

julia> max_error_norm(A, B, x, lambda)
9.506534782476192e-5

julia> lambda, x = lobpcg(A, X0, false, tol=tol, maxiter=Inf);

julia> max_error_norm(A, x, lambda)
9.83179424893794e-5

julia> @benchmark lobpcg($A, $B, $X0, false, tol=$tol, maxiter=Inf)
BenchmarkTools.Trial:
  memory estimate:  75.42 MiB
  allocs estimate:  26480
  --------------
  minimum time:     377.843 ms (1.23% GC)
  median time:      440.706 ms (1.63% GC)
  mean time:        452.468 ms (2.59% GC)
  maximum time:     645.701 ms (1.26% GC)
  --------------
  samples:          12
  evals/sample:     1

julia> @benchmark eigs($A, $B, nev=2, which=:SM, tol=$tol)
BenchmarkTools.Trial:
  memory estimate:  736.38 MiB
  allocs estimate:  10453
  --------------
  minimum time:     9.353 s (0.81% GC)
  median time:      9.353 s (0.81% GC)
  mean time:        9.353 s (0.81% GC)
  maximum time:     9.353 s (0.81% GC)
  --------------
  samples:          1
  evals/sample:     1

julia> @benchmark lobpcg($A, $B, $X0, true, tol=$tol, maxiter=Inf)
BenchmarkTools.Trial:
  memory estimate:  71.58 MiB
  allocs estimate:  21397
  --------------
  minimum time:     300.412 ms (2.28% GC)
  median time:      384.011 ms (1.55% GC)
  mean time:        396.732 ms (2.76% GC)
  maximum time:     573.312 ms (0.89% GC)
  --------------
  samples:          13
  evals/sample:     1

julia> @benchmark eigs($A, $B, nev=2, which=:LM, tol=$tol)
BenchmarkTools.Trial:
  memory estimate:  681.50 MiB
  allocs estimate:  5308
  --------------
  minimum time:     6.355 s (0.99% GC)
  median time:      6.355 s (0.99% GC)
  mean time:        6.355 s (0.99% GC)
  maximum time:     6.355 s (0.99% GC)
  --------------
  samples:          1
  evals/sample:     1

julia> @benchmark lobpcg($A, $X0, false, tol=$tol, maxiter=Inf)
BenchmarkTools.Trial:
  memory estimate:  44.38 MiB
  allocs estimate:  20467
  --------------
  minimum time:     190.071 ms (0.77% GC)
  median time:      238.091 ms (1.42% GC)
  mean time:        248.487 ms (2.56% GC)
  maximum time:     318.179 ms (1.35% GC)
  --------------
  samples:          21
  evals/sample:     1

julia> @benchmark eigs($A, nev=2, which=:SM, tol=$tol)
BenchmarkTools.Trial:
  memory estimate:  668.22 MiB
  allocs estimate:  2570
  --------------
  minimum time:     7.031 s (0.91% GC)
  median time:      7.031 s (0.91% GC)
  mean time:        7.031 s (0.91% GC)
  maximum time:     7.031 s (0.91% GC)
  --------------
  samples:          1
  evals/sample:     1

```

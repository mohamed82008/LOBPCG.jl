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

julia> maxerr(A, B, X, lambda) = maximum(sum((x)->(conj(x)*x), A*X-B*X*diagm(lambda), 1))
maxerr (generic function with 1 method)

julia> maxerr(A, X, lambda) = maximum(sum((x)->(conj(x)*x), A*X-X*diagm(lambda), 1))
maxerr (generic function with 2 methods)

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

julia> lambda, X = lobpcg(A, B, X0, false, tol=tol);

julia> maxerr(A, B, X, lambda)
8.088281961209552e-8

julia> lambda, X = lobpcg(A, X0, false, tol=tol);

julia> maxerr(A, X, lambda)
9.523001324147468e-5

julia> @benchmark lobpcg($A, $B, $X0, false, tol=$tol)
BenchmarkTools.Trial:
memory estimate:  74.08 MiB
allocs estimate:  18846
--------------
minimum time:     290.257 ms (1.52% GC)
median time:      301.370 ms (2.26% GC)
mean time:        308.458 ms (3.29% GC)
maximum time:     380.284 ms (19.79% GC)
--------------
samples:          17
evals/sample:     1

julia> @benchmark eigs($A, $B, nev=2, which=:SM, tol=$tol)
BenchmarkTools.Trial:
memory estimate:  727.35 MiB
allocs estimate:  8563
--------------
minimum time:     8.065 s (0.87% GC)
median time:      8.065 s (0.87% GC)
mean time:        8.065 s (0.87% GC)
maximum time:     8.065 s (0.87% GC)
--------------
samples:          1
evals/sample:     1

julia> @benchmark lobpcg($A, $B, $X0, true, tol=$tol)
BenchmarkTools.Trial:
memory estimate:  65.67 MiB
allocs estimate:  18571
--------------
minimum time:     262.807 ms (2.26% GC)
median time:      292.508 ms (1.90% GC)
mean time:        294.680 ms (3.16% GC)
maximum time:     372.151 ms (20.32% GC)
--------------
samples:          17
evals/sample:     1

julia> @benchmark eigs($A, $B, nev=2, which=:LM, tol=$tol)
BenchmarkTools.Trial:
memory estimate:  681.42 MiB
allocs estimate:  4487
--------------
minimum time:     6.875 s (0.98% GC)
median time:      6.875 s (0.98% GC)
mean time:        6.875 s (0.98% GC)
maximum time:     6.875 s (0.98% GC)
--------------
samples:          1
evals/sample:     1

julia> @benchmark lobpcg($A, $X0, false, tol=$tol)
BenchmarkTools.Trial:
memory estimate:  45.01 MiB
allocs estimate:  17122
--------------
minimum time:     163.061 ms (0.91% GC)
median time:      167.357 ms (1.13% GC)
mean time:        168.513 ms (1.34% GC)
maximum time:     181.494 ms (2.32% GC)
--------------
samples:          30
evals/sample:     1

julia> @benchmark eigs($A, nev=2, which=:SM, tol=$tol)
BenchmarkTools.Trial:
memory estimate:  671.82 MiB
allocs estimate:  2213
--------------
minimum time:     5.793 s (1.06% GC)
median time:      5.793 s (1.06% GC)
mean time:        5.793 s (1.06% GC)
maximum time:     5.793 s (1.06% GC)
--------------
samples:          1
evals/sample:     1

```

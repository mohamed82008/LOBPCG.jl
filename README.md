# LOBPCG

[![Build Status](https://travis-ci.org/mohamed82008/LOBPCG.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/LOBPCG.jl)

[![Coverage Status](https://coveralls.io/repos/mohamed82008/LOBPCG.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mohamed82008/LOBPCG.jl?branch=master)

[![codecov.io](http://codecov.io/github/mohamed82008/LOBPCG.jl/coverage.svg?branch=master)](http://codecov.io/github/mohamed82008/LOBPCG.jl?branch=master)

## Introduction

This package implements the algorithm for finding minimum or maximum generalized eigenvalues and vectors from "Andrew V. Knyazev. Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate
Gradient Method. SIAM Journal on Scientific Computing, 23(2):517-541, 2001." The algorithm tries to maximize or minimize the Rayleigh quotient using an enhanced conjugate gradient method. This package is still in the proof of concept stage and only implements the single eigenvector version (i.e. no blocking).

## Example

```julia
julia> using LOBPCG

julia> using BenchmarkTools

julia> T = Float64;

julia> n = 100000;

julia> A = 10*sprand(T, n, n, 10/n);

julia> A = A' + A + 200I;

julia> B = 10*sprand(T, n, n, 10/n);

julia> B = B' + B + 200I;

julia> a = zeros(T, n);

julia> tol = T(1e-4)
0.0001

julia> lambda, x = locg(A, B, a, Val{:Max}, tol);

julia> norm(A*x - lambda * B * x)
9.947322586451714e-5

julia> buffers = LOCGBuffersSimple(a);

julia> @time lambda, x = locg(A, buffers, Val{:Min}, tol);
  1.300320 seconds (171.74 k allocations: 6.836 MiB)

julia> buffers = LOCGBuffersSimple(a);

julia> @time lambda, x = locg(A, buffers, Val{:Min}, tol);
  0.294669 seconds (9 allocations: 288 bytes)

julia> norm(A * x - lambda * x)
8.442852486067891e-5

julia> _I = speye(n,n);

julia> buffers = LOCGBuffersGeneral(a);

julia> @time lambda, x = locg(_I, B, buffers, Val{:Max}, tol);
  0.904644 seconds (163 allocations: 9.991 KiB)

julia> buffers = LOCGBuffersGeneral(a);

julia> @time lambda, x = locg(_I, B, buffers, Val{:Max}, tol);
  0.558346 seconds (9 allocations: 288 bytes)

julia> @benchmark locg($A, $a, Val{:Max}, $tol)
BenchmarkTools.Trial:
  memory estimate:  9.92 MiB
  allocs estimate:  20
  --------------
  minimum time:     88.074 ms (0.00% GC)
  median time:      91.625 ms (0.00% GC)
  mean time:        91.905 ms (0.84% GC)
  maximum time:     96.656 ms (1.90% GC)
  --------------
  samples:          55
  evals/sample:     1

julia> @benchmark eigs($A, nev=1, which=:LM, tol=$tol)
BenchmarkTools.Trial:
  memory estimate:  19.85 MiB
  allocs estimate:  250
  --------------
  minimum time:     222.059 ms (0.00% GC)
  median time:      228.296 ms (1.11% GC)
  mean time:        228.548 ms (1.09% GC)
  maximum time:     234.640 ms (1.40% GC)
  --------------
  samples:          22
  evals/sample:     1

julia> @benchmark locg($A, $a, Val{:Min}, $tol)
BenchmarkTools.Trial:
  memory estimate:  9.92 MiB
  allocs estimate:  20
  --------------
  minimum time:     278.934 ms (0.82% GC)
  median time:      295.056 ms (0.00% GC)
  mean time:        593.186 ms (0.15% GC)
  maximum time:     1.886 s (0.10% GC)
  --------------
  samples:          9
  evals/sample:     1

julia> @benchmark eigs($A, nev=1, which=:SR, tol=$tol)
BenchmarkTools.Trial:
  memory estimate:  19.86 MiB
  allocs estimate:  740
  --------------
  minimum time:     1.002 s (0.26% GC)
  median time:      1.014 s (0.26% GC)
  mean time:        1.152 s (0.17% GC)
  maximum time:     1.608 s (0.28% GC)
  --------------
  samples:          5
  evals/sample:     1

```
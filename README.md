# LOBPCG

[![Build Status](https://travis-ci.org/mohamed82008/LOBPCG.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/LOBPCG.jl)

[![Coverage Status](https://coveralls.io/repos/mohamed82008/LOBPCG.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mohamed82008/LOBPCG.jl?branch=master)

[![codecov.io](http://codecov.io/github/mohamed82008/LOBPCG.jl/coverage.svg?branch=master)](http://codecov.io/github/mohamed82008/LOBPCG.jl?branch=master)

## Introduction

This package implements the algorithm for finding minimum or maximum generalized eigenvalues and vectors from "Andrew V. Knyazev. Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate
Gradient Method. SIAM Journal on Scientic Computing, 23(2):517541, 2001." The algorithm tries to maximize or minimize the Rayleigh quotient using an enhanced conjugate gradient method. This package is still in the proof of concept stage and only implements the single eigenvector version (i.e. no blocking).

## Example

```julia
julia> using LOBPCG

julia> A = sprand(100,100,0.1);

julia> A = A' * A + I;

julia> a = zeros(100);

julia> lambda1, xmin1 = locg(A, I, a, Val{:Min}, 1e-6);

julia> _lambda2, _xmin2 = eigs(A, nev=1, which=:SM);

julia> lambda2 = _lambda2[1]; xmin2 = reshape(_xmin2, size(_xmin2,1));

julia> lambda1 ≈ lambda2
true

julia> norm(xmin1-xmin2)
6.966134097192094e-5

julia> lambda1, xmax1 = locg(A, I, a, Val{:Max}, 1e-6);

julia> _lambda2, _xmax2 = eigs(A, nev=1, which=:LM);

julia> lambda2 = _lambda2[1]; xmax2 = reshape(_xmax2, size(_xmax2,1));

julia> lambda1 ≈ lambda2
true

julia> norm(xmax1-xmax2)
1.6487634983105846e-8

```
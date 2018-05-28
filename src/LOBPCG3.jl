#=
The code below was derived from the scipy implementation of the LOBPCG algorithm in https://github.com/scipy/scipy/blob/v1.1.0/scipy/sparse/linalg/eigen/lobpcg/lobpcg.py#L109-L568. 

Since the link above mentions the license is BSD license, the notice for the BSD license 2.0 is hereby given below giving credit to the authors of the Python implementation.

Copyright (c) 2018, Robert Cimrman, Andrew Knyazev
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

struct Blocks{Generalized, T, TA<:AbstractArray{T}}
    block::TA # X, R or P
    A_block::TA # AX, AR or AP
    B_block::TA # BX, BR or BP
end
Blocks(X, AX) = Blocks{false, eltype(X), typeof(X)}(X, AX, X)
Blocks(X, AX, BX) = Blocks{true, eltype(X), typeof(X)}(X, AX, BX)
function AB_mul_X!(b::Blocks{false}, A, B)
    A_mul_B!(b.A_block, A, b.block)
    return
end
function AB_mul_X!(b::Blocks{false}, A, B, n)
    A_mul_B!(view(b.A_block, :, 1:n), A, view(b.block, :, 1:n))
    return
end
function A_mul_X!(b::Blocks, A)
    A_mul_B!(b.A_block, A, b.block)
    return
end
function A_mul_X!(b::Blocks, A, n)
    A_mul_B!(view(b.A_block, :, 1:n), A, view(b.block, :, 1:n))
    return
end
function B_mul_X!(b::Blocks{true}, B)
    A_mul_B!(b.B_block, B, b.block)
    return
end
function B_mul_X!(b::Blocks{true}, B, n)
    A_mul_B!(view(b.B_block, :, 1:n), B, view(b.block, :, 1:n))
    return
end
function B_mul_X!(b::Blocks{false}, B, n = 0)
    return
end
function AB_mul_X!(b::Blocks{true}, A, B)
    A_mul_B!(b.A_block, A, b.block)
    A_mul_B!(b.B_block, B, b.block)
    return
end
function AB_mul_X!(b::Blocks{true}, A, B, n)
    A_mul_B!(view(b.A_block, :, 1:n), A, view(b.block, :, 1:n))
    A_mul_B!(view(b.B_block, :, 1:n), B, view(b.block, :, 1:n))
    return
end

struct Constraint{T, TA<:AbstractArray{T}, TC}
    Y::TA
    BY::TA
    gram_chol::TC
    gramYBV::TA # to be used in view
    tmp::TA # to be used in view
end
function Constraint(::Void, B, X)
    return Constraint{Void, Matrix{Void}, Void}(Matrix{Void}(0,0), Matrix{Void}(0,0), nothing, Matrix{Void}(0,0), Matrix{Void}(0,0))
end
function Constraint(Y, B, X)
    T = eltype(X)
    if B isa Void
        B = Y
    else
        BY = similar(B)
        A_mul_B!(BY, B, Y)
    end
    gramYBY = At_mul_B(Y, BY)
    gramYBY_chol = cholfact!(Hermitian(gramYBY))
    gramYBV = zeros(T, size(Y, 2), size(X, 2))
    tmp = similar(gramYBV)

    return Constraint(Y, BY, gramYBY_chol, gramYBV, tmp)
end

function (constr!::Constraint{Void})(X)
    nothing
end

function (constr!::Constraint)(X)
    sizeX = size(X, 2)
    sizeY = size(constr!.Y, 2)
    gramYBV_view = view(constr!.gramYBV, 1:sizeY, 1:sizeX)
    At_mul_B!(gramYBV_view, constr!.BY, X)
    tmp_view = view(constr!.tmp, 1:sizeY, 1:sizeX)
    A_ldiv_B!(tmp_view, gram_chol, gramYBV_view)
    A_mul_B!(X, constr!.Y, tmp_view)

    nothing
end
Base.size(::Constraint{Void}) = 0
Base.size(c::Constraint) = size(c.Y, 2)

struct RPreconditioner{TM, T, TA<:AbstractArray{T}}
    M::TM
    buffer::TA
    RPreconditioner{TM, T, TA}(M, X) where {TM, T, TA<:AbstractArray{T}} = new(M, similar(X))
end
RPreconditioner(M, X) = RPreconditioner{typeof(M), eltype(X), typeof(X)}(M, X)

function (precond!::RPreconditioner{Void})(X)
    nothing
end
function (precond!::RPreconditioner)(X)
    A_mul_B!(precond!.buffer, precond!.M, X)
    # Just returning buffer would be cheaper but struct at call site must be mutable
    X .= precond!.buffer
    nothing
end

struct BlockGram{Generalized, TA}
    XAX::TA
    XAR::TA
    XAP::TA
    RAR::TA
    RAP::TA
    PAP::TA
end
function BlockGram(XBlocks::Blocks{Generalized, T}) where {Generalized, T}
    sizeX = size(XBlocks.block, 2)
    XAX = zeros(T, sizeX, sizeX)
    XAP = zeros(T, sizeX, sizeX)
    XAR = zeros(T, sizeX, sizeX)
    RAR = zeros(T, sizeX, sizeX)
    RAP = zeros(T, sizeX, sizeX)
    PAP = zeros(T, sizeX, sizeX)
    return BlockGram{Generalized, Matrix{T}}(XAX, XAR, XAP, RAR, RAP, PAP)
end
XAX!(BlockGram, XBlocks) = At_mul_B!(BlockGram.XAX, XBlocks.block, XBlocks.A_block)
XAP!(BlockGram, XBlocks, PBlocks, n) = At_mul_B!(view(BlockGram.XAP, :, 1:n), XBlocks.block, view(PBlocks.A_block, :, 1:n))
XAR!(BlockGram, XBlocks, RBlocks, n) = At_mul_B!(view(BlockGram.XAR, :, 1:n), XBlocks.block, view(RBlocks.A_block, :, 1:n))
RAR!(BlockGram, RBlocks, n) = At_mul_B!(view(BlockGram.RAR, 1:n, 1:n), view(RBlocks.block, :, 1:n), view(RBlocks.A_block, :, 1:n))
RAP!(BlockGram, RBlocks, PBlocks, n) = At_mul_B!(view(BlockGram.RAP, 1:n, 1:n), view(RBlocks.A_block, :, 1:n), view(PBlocks.block, :, 1:n))
PAP!(BlockGram, PBlocks, n) = At_mul_B!(view(BlockGram.PAP, 1:n, 1:n), view(PBlocks.block, :, 1:n), view(PBlocks.A_block, :, 1:n))
XBP!(BlockGram, XBlocks, PBlocks, n) = At_mul_B!(view(BlockGram.XAP, :, 1:n), XBlocks.block, view(PBlocks.B_block, :, 1:n))
XBR!(BlockGram, XBlocks, RBlocks, n) = At_mul_B!(view(BlockGram.XAR, :, 1:n), XBlocks.block, view(RBlocks.B_block, :, 1:n))
RBP!(BlockGram, RBlocks, PBlocks, n) = At_mul_B!(view(BlockGram.RAP, 1:n, 1:n), view(RBlocks.B_block, :, 1:n), view(PBlocks.block, :, 1:n))
XBX!(BlockGram, XBlocks) = At_mul_B!(BlockGram.XAX, XBlocks.block, XBlocks.B_block)
RBR!(BlockGram, RBlocks, n) = At_mul_B!(view(BlockGram.RAR, 1:n, 1:n), view(RBlocks.block, :, 1:n), view(RBlocks.B_block, :, 1:n))
PBP!(BlockGram, PBlocks, n) = At_mul_B!(view(BlockGram.PAP, 1:n, 1:n), view(PBlocks.block, :, 1:n), view(PBlocks.B_block, :, 1:n))

function (g::BlockGram)(gram, lambda, n1::Int, n2::Int, n3::Int)
    xr = 1:n1
    rr = n1+1:n1+n2
    pr = n1+n2+1:n1+n2+n3
    if n1 > 0
        #gram[xr, xr] .= view(g.XAX, 1:n1, 1:n1)
        gram[xr, xr] .= Diagonal(view(lambda, 1:n1))
    end
    if n2 > 0
        gram[rr, rr] .= view(g.RAR, 1:n2, 1:n2)
        gram[xr, rr] .= view(g.XAR, 1:n1, 1:n2)
        transpose!(view(gram, rr, xr), view(g.XAR, 1:n1, 1:n2))
    end
    if n3 > 0
        gram[pr, pr] .= view(g.PAP, 1:n3, 1:n3)
        gram[rr, pr] .= view(g.RAP, 1:n2, 1:n3)
        gram[xr, pr] .= view(g.XAP, 1:n1, 1:n3)
        transpose!(view(gram, pr, rr), view(g.RAP, 1:n2, 1:n3))
        transpose!(view(gram, pr, xr), view(g.XAP, 1:n1, 1:n3))
    end
    return 
end
function (g::BlockGram)(gram, n1::Int, n2::Int, n3::Int, normalized::Bool)
    xr = 1:n1
    rr = n1+1:n1+n2
    pr = n1+n2+1:n1+n2+n3
    if n1 > 0
        if normalized
            for j in xr, i in xr
                gram[i, j] = ifelse(i==j, 1, 0)
            end
        else
            gram[xr, xr] .= view(g.XAX, 1:n1, 1:n1)
        end
    end
    if n2 > 0
        if normalized
            for j in rr, i in rr
                gram[i, j] = ifelse(i==j, 1, 0)
            end
        else
            gram[rr, rr] .= view(g.RAR, 1:n2, 1:n2)
        end
        gram[xr, rr] .= view(g.XAR, 1:n1, 1:n2)
        transpose!(view(gram, rr, xr), view(g.XAR, 1:n1, 1:n2))
    end
    if n3 > 0
        if normalized
            for j in pr, i in pr
                gram[i, j] = ifelse(i==j, 1, 0)
            end
        else
            gram[pr, pr] .= view(g.PAP, 1:n3, 1:n3)
        end
        gram[rr, pr] .= view(g.RAP, 1:n2, 1:n3)
        gram[xr, pr] .= view(g.XAP, 1:n1, 1:n3)
        transpose!(view(gram, pr, rr), view(g.RAP, 1:n2, 1:n3))
        transpose!(view(gram, pr, xr), view(g.XAP, 1:n1, 1:n3))
    end
    return 
end

abstract type AbstractOrtho end
struct CholQR{TA} <: AbstractOrtho
    gramVBV::TA # to be used in view
end
function (ortho!::CholQR)(XBlocks::Blocks{Generalized}; update_AX=false, update_BX=false) where Generalized
    X = XBlocks.block
    BX = XBlocks.B_block # Assumes it is premultiplied
    sizeX = size(X, 2)
    gram_view = view(ortho!.gramVBV, 1:sizeX, 1:sizeX)
    At_mul_B!(gram_view, X, BX)
    cholf = cholfact!(Hermitian(gram_view))
    R = cholf.factors
    X[:,1] .= view(X, :, 1) ./ R[1,1]
    for i in 2:sizeX
        for j in 1:i-1
            X[:,i] .= view(X, :, i) - view(X, :, j) .* R[j,i]
        end
        X[:,i] .= view(X, :, i) ./ R[i,i]
    end

    if update_AX
        X = XBlocks.A_block
        X[:,1] .= view(X, :, 1) ./ R[1,1]
        for i in 2:sizeX
            for j in 1:i-1
                X[:,i] .= view(X, :, i) - view(X, :, j) .* R[j,i]
            end
            X[:,i] .= view(X, :, i) ./ R[i,i]
        end    
    end

    if Generalized
        if update_BX
            X = XBlocks.B_block
            X[:,1] .= view(X, :, 1) ./ R[1,1]
            for i in 2:sizeX
                for j in 1:i-1
                    X[:,i] .= view(X, :, i) - view(X, :, j) .* R[j,i]
                end
                X[:,i] .= view(X, :, i) ./ R[i,i]
            end    
        end
    end
    return 
end
function (ortho!::CholQR)(XBlocks::Blocks{Generalized}, sizeX; update_AX=false, update_BX=false) where Generalized
    X = XBlocks.block
    BX = XBlocks.B_block # Assumes it is premultiplied
    gram_view = view(ortho!.gramVBV, 1:sizeX, 1:sizeX)
    At_mul_B!(gram_view, view(X, :, 1:sizeX), view(BX, :, 1:sizeX))
    cholf = cholfact!(Hermitian(gram_view))
    R = cholf.factors
    X[:,1] .= view(X, :, 1) ./ R[1,1]
    for i in 2:sizeX
        for j in 1:i-1
            X[:,i] .= view(X, :, i) - view(X, :, j) .* R[j,i]
        end
        X[:,i] .= view(X, :, i) ./ R[i,i]
    end    

    if update_AX
        X = XBlocks.A_block
        X[:,1] .= view(X, :, 1) ./ R[1,1]
        for i in 2:sizeX
            for j in 1:i-1
                X[:,i] .= view(X, :, i) - view(X, :, j) .* R[j,i]
            end
            X[:,i] .= view(X, :, i) ./ R[i,i]
        end    
    end

    if Generalized
        if update_BX
            X = XBlocks.B_block
            X[:,1] .= view(X, :, 1) ./ R[1,1]
            for i in 2:sizeX
                for j in 1:i-1
                    X[:,i] .= view(X, :, i) - view(X, :, j) .* R[j,i]
                end
                X[:,i] .= view(X, :, i) ./ R[i,i]
            end    
        end
    end

    return 
end

struct RayleighRitz{Generalized, T, TA, TB, TL<:AbstractVector{T}, TVec<:AbstractVector{Int}, TV<:AbstractArray{T}, TBlocks<:Blocks{Generalized, T}, TO<:AbstractOrtho, TP, TC, TG, TH, TM}
    A::TA
    B::TB
    λ::TL
    λperm::TVec
    V::TV
    residuals::TL
    largest::Bool
    XBlocks::TBlocks
    tempXBlocks::TBlocks
    PBlocks::TBlocks
    activePBlocks::TBlocks # to be used in view
    RBlocks::TBlocks
    activeRBlocks::TBlocks # to be used in view
    iteration::Base.RefValue{Int}
    currentBlockSize::Base.RefValue{Int}
    ortho!::TO
    precond!::TP
    constr!::TC
    gramABlock::TG
    gramBBlock::TG
    gramA::TV
    gramB::TV
    residualNormsHistory::TH
    activeMask::TM
end
function RayleighRitz(A, B, M, Y, X, largest)
    T = eltype(X)
    constr! = Constraint(Y, B, X)
    precond! = RPreconditioner(M, X)

    nev = size(X, 2)
    if B isa Void
        XBlocks = Blocks(copy(X), similar(X))
        tempXBlocks = Blocks(copy(X), similar(X))
        RBlocks = Blocks(similar(X), similar(X))
        activeRBlocks = Blocks(similar(X), similar(X))
        PBlocks = Blocks(similar(X), similar(X))
        activePBlocks = Blocks(similar(X), similar(X))
    else
        XBlocks = Blocks(copy(X), similar(X), similar(X))
        tempXBlocks = Blocks(copy(X), similar(X), similar(X))
        RBlocks = Blocks(similar(X), similar(X), similar(X))
        activeRBlocks = Blocks(similar(X), similar(X), similar(X))
        PBlocks = Blocks(similar(X), similar(X), similar(X))
        activePBlocks = Blocks(similar(X), similar(X), similar(X))
    end
    λ = zeros(T, nev*3)
    λperm = zeros(Int, nev*3)
    V = zeros(T, nev*3, nev*3)
    residuals = zeros(T, nev)
    iteration = Ref(1)
    currentBlockSize = Ref(nev)
    generalized = !(B isa Void)
    ortho! = CholQR(zeros(T, nev, nev))

    gramABlock = BlockGram(XBlocks)
    gramBBlock = BlockGram(XBlocks)

    residualNormsHistory = Vector{Float64}[]
    activeMask = ones(Bool, nev)

    gramA = zeros(T, 3*nev, 3*nev)
    gramB = zeros(T, 3*nev, 3*nev)

    return RayleighRitz{generalized, T, typeof(A), typeof(B), typeof(λ), typeof(λperm), typeof(V), typeof(XBlocks), typeof(ortho!), typeof(precond!), typeof(constr!), typeof(gramABlock), typeof(residualNormsHistory), typeof(activeMask)}(A, B, λ, λperm, V, residuals, largest, XBlocks, tempXBlocks, PBlocks, activePBlocks, RBlocks, activeRBlocks, iteration, currentBlockSize, ortho!, precond!, constr!, gramABlock, gramBBlock, gramA, gramB, residualNormsHistory, activeMask)
end

function (rr::RayleighRitz{Generalized})(residualTolerance) where Generalized
    sizeX = size(rr.XBlocks.block, 2)
    if rr.iteration[] == 1
        # Finds BX
        # This is important before calling ortho! because ortho! uses BX
        B_mul_X!(rr.XBlocks, rr.B)
        
        # Orthonormalizes X and updates BX
        rr.ortho!(rr.XBlocks, update_BX=true)

        # Updates AX
        A_mul_X!(rr.XBlocks, rr.A)

        # Finds gram matrix X'AX
        XAX!(rr.gramABlock, rr.XBlocks)

        # Solve the Rayleigh-Ritz sub-problem
        eigf = eigfact!(Hermitian(rr.gramABlock.XAX))

        # Selects extremal eigenvalues and corresponding vectors
        selectperm!(view(rr.λperm, 1:sizeX), eigf.values, 1:sizeX, rev=rr.largest)
        rr.λ[1:sizeX] .= view(eigf.values, view(rr.λperm, 1:sizeX))
        rr.V[1:sizeX, 1:sizeX] .= view(eigf.vectors, :, view(rr.λperm, 1:sizeX))

        # Updates Ritz vectors X and updates AX and BX accordingly
        x_eigview = view(rr.V, 1:sizeX, 1:sizeX)
        A_mul_B!(rr.tempXBlocks.block, rr.XBlocks.block, x_eigview)
        rr.XBlocks.block .= rr.tempXBlocks.block        
        A_mul_B!(rr.tempXBlocks.A_block, rr.XBlocks.A_block, x_eigview)
        rr.XBlocks.A_block .= rr.tempXBlocks.A_block
        if Generalized
            A_mul_B!(rr.tempXBlocks.B_block, rr.XBlocks.B_block, x_eigview)
            rr.XBlocks.B_block .= rr.tempXBlocks.B_block
        end
    elseif rr.iteration[] == 2
        # Finds residual vectors
        A_mul_B!(rr.RBlocks.block, rr.XBlocks.B_block, Diagonal(view(rr.λ, 1:sizeX)))
        rr.RBlocks.block .= rr.XBlocks.A_block .- rr.RBlocks.block
        # Finds residual norms
        for j in 1:size(rr.RBlocks.block, 2)
            rr.residuals[j] = 0
            for i in 1:size(rr.RBlocks.block, 1)
                x = rr.RBlocks.block[i,j]
                rr.residuals[j] += conj(x)*x
            end
            rr.residuals[j] = sqrt(rr.residuals[j])
        end
        # Store history of norms
        push!(rr.residualNormsHistory, rr.residuals[1:sizeX])

        # Update active vectors mask
        rr.activeMask .*= view(rr.residuals, 1:sizeX) .> residualTolerance
        rr.currentBlockSize[] = sum(rr.activeMask)
        rr.currentBlockSize[] == 0 && return 

        bs = rr.currentBlockSize[]
        # Update active R blocks
        rr.activeRBlocks.block[:, 1:bs] .= view(rr.RBlocks.block, :, rr.activeMask)

        # Precondition the active residual vectors
        rr.precond!(view(rr.activeRBlocks.block, 1:bs))
        
        # Constrain the active residual vectors to be B-orthogonal to Y
        rr.constr!(view(rr.activeRBlocks.block, 1:bs))

        # Find BR for the active residuals
        B_mul_X!(rr.activeRBlocks, rr.B, bs)

        # Orthonormalize the active residuals and updates BR
        rr.ortho!(rr.activeRBlocks, bs, update_BX=true)

        # Finds AR for the active residuals
        A_mul_X!(rr.activeRBlocks, rr.A, bs)

        # Find X'AX, R'AR and X'AR
        XAX!(rr.gramABlock, rr.XBlocks)
        XAR!(rr.gramABlock, rr.XBlocks, rr.activeRBlocks, bs)
        RAR!(rr.gramABlock, rr.activeRBlocks, bs)
        # Find X'BR
        XBR!(rr.gramBBlock, rr.XBlocks, rr.activeRBlocks, bs)

        # Update the gram matrix [X R]' A [X R]
        rr.gramABlock(rr.gramA, view(rr.λ, 1:sizeX), sizeX, bs, 0)
        
        # Update the gram matrix [X R]' B [X R]
        rr.gramBBlock(rr.gramB, sizeX, bs, 0, true)

        # Solve the Rayleigh-Ritz sub-problem
        subdim = sizeX+bs
        gramAview = view(rr.gramA, 1:subdim, 1:subdim)
        gramBview = view(rr.gramB, 1:subdim, 1:subdim)
        eigf = eigfact!(Hermitian(gramAview), Hermitian(gramBview))

        # Selects extremal eigenvalues and corresponding vectors
        selectperm!(view(rr.λperm, 1:subdim), eigf.values, 1:subdim, rev=rr.largest)
        rr.λ[1:sizeX] .= view(eigf.values, view(rr.λperm, 1:sizeX))
        rr.V[1:sizeX+bs, 1:sizeX] .= view(eigf.vectors, :, view(rr.λperm, 1:sizeX))

        # Updates Ritz vectors X and updates AX and BX accordingly
        # And updates P, AP and BP
        A_mul_B!(rr.tempXBlocks.block, rr.XBlocks.block, view(rr.V, 1:sizeX, 1:sizeX))

        A_mul_B!(rr.tempXBlocks.A_block, rr.XBlocks.A_block, view(rr.V, 1:sizeX, 1:sizeX))

        if Generalized
            A_mul_B!(rr.tempXBlocks.B_block, rr.XBlocks.B_block, view(rr.V, 1:sizeX, 1:sizeX))
        end

        A_mul_B!(rr.PBlocks.block, view(rr.activeRBlocks.block, :, 1:bs), view(rr.V, sizeX+1:sizeX+bs, 1:sizeX))

        A_mul_B!(rr.PBlocks.A_block, view(rr.activeRBlocks.A_block, :, 1:bs), view(rr.V, sizeX+1:sizeX+bs, 1:sizeX))

        if Generalized
            A_mul_B!(rr.PBlocks.B_block, view(rr.activeRBlocks.B_block, :, 1:bs), view(rr.V, sizeX+1:sizeX+bs, 1:sizeX))
        end

        rr.XBlocks.block .= rr.tempXBlocks.block .+ rr.PBlocks.block
        rr.XBlocks.A_block .= rr.tempXBlocks.A_block .+ rr.PBlocks.A_block
        if Generalized
            rr.XBlocks.B_block .= rr.tempXBlocks.B_block .+ rr.PBlocks.B_block
        end
    else
        # Finds residual vectors
        A_mul_B!(rr.RBlocks.block, rr.XBlocks.B_block, Diagonal(view(rr.λ, 1:sizeX)))
        rr.RBlocks.block .= rr.XBlocks.A_block .- rr.RBlocks.block
        # Finds residual norms
        for j in 1:size(rr.RBlocks.block, 2)
            rr.residuals[j] = 0
            for i in 1:size(rr.RBlocks.block, 1)
                x = rr.RBlocks.block[i,j]
                rr.residuals[j] += conj(x)*x
            end
            rr.residuals[j] = sqrt(rr.residuals[j])
        end
        # Store history of norms
        push!(rr.residualNormsHistory, rr.residuals[1:sizeX])

        # Update active vectors mask
        rr.activeMask .*= view(rr.residuals, 1:sizeX) .> residualTolerance
        rr.currentBlockSize[] = sum(rr.activeMask)
        rr.currentBlockSize[] == 0 && return

        # Update active blocks
        bs = rr.currentBlockSize[]

        rr.activeRBlocks.block[:, 1:bs] .= view(rr.RBlocks.block, :, rr.activeMask)
        
        rr.activePBlocks.block[:, 1:bs] .= view(rr.PBlocks.block, :, rr.activeMask)
        rr.activePBlocks.A_block[:, 1:bs] .= view(rr.PBlocks.A_block, :, rr.activeMask)
        rr.activePBlocks.B_block[:, 1:bs] .= view(rr.PBlocks.B_block, :, rr.activeMask)

        # Precondition the active residual vectors
        rr.precond!(view(rr.activeRBlocks.block, 1:bs))
        
        # Constrain the active residual vectors to be B-orthogonal to Y
        rr.constr!(view(rr.activeRBlocks.block, 1:bs))

        # Find BR for the active residuals
        B_mul_X!(rr.activeRBlocks, rr.B, bs)

        # Orthonormalize the active residuals, and update BR accordingly
        rr.ortho!(rr.activeRBlocks, bs, update_BX=true)

        # Updates AR for the active residuals
        A_mul_X!(rr.activeRBlocks, rr.A, bs)

        # Orthonormalizes P and updates AP
        # Probably not be necessary to update BX and AX here
        rr.ortho!(rr.activePBlocks, bs, update_AX=true, update_BX=true)

        # Find X'AX, R'AR, P'AP, X'AR, X'AP and R'AP
        XAX!(rr.gramABlock, rr.XBlocks)
        XAR!(rr.gramABlock, rr.XBlocks, rr.activeRBlocks, bs)
        XAP!(rr.gramABlock, rr.XBlocks, rr.activePBlocks, bs)
        RAR!(rr.gramABlock, rr.activeRBlocks, bs)
        RAP!(rr.gramABlock, rr.activeRBlocks, rr.activePBlocks, bs)
        PAP!(rr.gramABlock, rr.activePBlocks, bs)        

        # Update the gram matrix [X R P]' A [X R P]
        rr.gramABlock(rr.gramA, view(rr.λ, 1:sizeX), sizeX, bs, bs)

        # Can be removed
        #XBX!(rr.gramBBlock, rr.XBlocks)
        #RBR!(rr.gramBBlock, rr.activeRBlocks, bs)
        #PBP!(rr.gramBBlock, rr.activePBlocks, bs)

        # Find X'BR, X'BP and P'BR
        XBR!(rr.gramBBlock, rr.XBlocks, rr.activeRBlocks, bs)
        XBP!(rr.gramBBlock, rr.XBlocks, rr.activePBlocks, bs)
        RBP!(rr.gramBBlock, rr.activeRBlocks, rr.activePBlocks, bs)
        
        # Update the gram matrix [X R P]' B [X R P]
        rr.gramBBlock(rr.gramB, sizeX, bs, bs, true)
        
        # Solve the Rayleigh-Ritz sub-problem
        subdim = sizeX + 2*bs
        gramAview = view(rr.gramA, 1:subdim, 1:subdim)
        gramBview = view(rr.gramB, 1:subdim, 1:subdim)
        eigf = eigfact!(Hermitian(gramAview), Hermitian(gramBview))

        # Selects extremal eigenvalues and corresponding vectors
        selectperm!(view(rr.λperm, 1:subdim), eigf.values, 1:subdim, rev=rr.largest)
        rr.λ[1:sizeX] .= view(eigf.values, view(rr.λperm, 1:sizeX))
        rr.V[1:subdim, 1:sizeX] .= view(eigf.vectors, :, view(rr.λperm, 1:sizeX))

        # Updates Ritz vectors X and updates AX and BX accordingly
        # And updates P, AP and BP
        r_eigview = view(rr.V, sizeX+1:sizeX+bs, 1:sizeX)
        p_eigview = view(rr.V, sizeX+bs+1:sizeX+2*bs, 1:sizeX)

        r_blockview = view(rr.activeRBlocks.block, :, 1:bs)
        p_blockview = view(rr.activePBlocks.block, :, 1:bs)
        A_mul_B!(rr.PBlocks.block, r_blockview , r_eigview)
        A_mul_B!(rr.tempXBlocks.block, p_blockview, p_eigview)
        rr.PBlocks.block .= rr.PBlocks.block .+ rr.tempXBlocks.block

        ra_blockview = view(rr.activeRBlocks.A_block, :, 1:bs)
        pa_blockview = view(rr.activePBlocks.A_block, :, 1:bs)
        A_mul_B!(rr.PBlocks.A_block, ra_blockview, r_eigview)
        A_mul_B!(rr.tempXBlocks.A_block, pa_blockview, p_eigview)
        rr.PBlocks.A_block .= rr.PBlocks.A_block .+ rr.tempXBlocks.A_block

        if Generalized
            rb_blockview = view(rr.activeRBlocks.B_block, :, 1:bs)
            pb_blockview = view(rr.activePBlocks.B_block, :, 1:bs)
            A_mul_B!(rr.PBlocks.B_block, rb_blockview, r_eigview)
            A_mul_B!(rr.tempXBlocks.B_block, pb_blockview, p_eigview)
            rr.PBlocks.B_block .= rr.PBlocks.B_block .+ rr.tempXBlocks.B_block
        end

        x_eigview = view(rr.V, 1:sizeX, 1:sizeX)
        A_mul_B!(rr.tempXBlocks.block, rr.XBlocks.block, x_eigview)
        rr.XBlocks.block .= rr.tempXBlocks.block .+ rr.PBlocks.block
        A_mul_B!(rr.tempXBlocks.A_block, rr.XBlocks.A_block, x_eigview)
        rr.XBlocks.A_block .= rr.tempXBlocks.A_block .+ rr.PBlocks.A_block
        if Generalized
            A_mul_B!(rr.tempXBlocks.B_block, rr.XBlocks.B_block, x_eigview)
            rr.XBlocks.B_block .= rr.tempXBlocks.B_block .+ rr.PBlocks.B_block
        end
    end

    return
end

function dense_solver(A, B, X, largest)
    warn("The problem size is small compared to the block size. Using dense eigensolver instead of LOBPCG.")
    # Define the closed range of indices of eigenvalues to return.
    n, sizeX = size(X)
    eigvals = largest ? (n - sizeX + 1, n) : (1, sizeX)
    A_dense = A * eye(n)
    B_dense = get_b_dense(B, n)
    return _eig(A_dense, B_dense)
end

"""Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)"""
function lobpcg(A, X::AbstractMatrix, largest::Bool=true, ::Type{Val{residualhistory}} = Val{false}; preconditioner=nothing, constraint=nothing, maxiter::Integer=100, tol::Number=nothing) where {residualhistory}
    lobpcg(A, nothing, X, largest, Val{residualhistory}; tol=tol, maxiter=maxiter, preconditioner=preconditioner, constraint=constraint)
end

function lobpcg(A, B, X, largest=true, ::Type{Val{residualhistory}}=Val{false};
                preconditioner=nothing, constraint=nothing, 
                tol=nothing, maxiter=100) where {residualhistory} 

    T = eltype(X)
    M = preconditioner
    Y = constraint
    n, sizeX = size(X)
    if Y isa Void
        if n < 5 * sizeX 
            return dense_solver(A, B, X,largest)
        end
    else
        sizeY = size(Y, 2)
        if (n - sizeY) < 5 * sizeX
            throw("The dense eigensolver does not support constraints.")
        end
    end
    if sizeX > n
        throw("X column dimension exceeds the row dimension")
    end

    for j in 1:size(X,2)
        if all(x -> x==0, view(X, :, j))
            X[:,j] .= rand.()
        end
    end
    rr = RayleighRitz(A, B, M, Y, X, largest)
    residualTolerance = (tol isa Void) ? sqrt(1e-15)*n : tol
    maxiter = min(n, maxiter)
    for iteration in 1:maxiter
        rr.iteration[] = iteration
        rr(residualTolerance)
        rr.currentBlockSize[] == 0 && break
    end
    if residualhistory
        return rr.λ[1:sizeX], rr.XBlocks.block, rr.residualNormsHistory
    else
        return rr.λ[1:sizeX], rr.XBlocks.block
    end
end

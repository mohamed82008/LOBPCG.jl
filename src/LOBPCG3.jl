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

function I!(G, xr)
    for j in xr, i in xr
        G[i, j] = ifelse(i==j, 1, 0)
    end
    return
end

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
function (g::BlockGram)(gram, n1::Int, n2::Int, n3::Int, normalized::Bool=true)
    xr = 1:n1
    rr = n1+1:n1+n2
    pr = n1+n2+1:n1+n2+n3
    if n1 > 0
        if normalized
            I!(gram, xr)
        else
            gram[xr, xr] .= view(g.XAX, 1:n1, 1:n1)
        end
    end
    if n2 > 0
        if normalized
            I!(gram, rr)
        else
            gram[rr, rr] .= view(g.RAR, 1:n2, 1:n2)
        end
        gram[xr, rr] .= view(g.XAR, 1:n1, 1:n2)
        transpose!(view(gram, rr, xr), view(g.XAR, 1:n1, 1:n2))
    end
    if n3 > 0
        if normalized
            I!(gram, pr)
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

function A_rdiv_B!(A, B::UpperTriangular)
    s = size(A, 2)
    A[:,1] .= view(A, :, 1) ./ B[1,1]
    for i in 2:s
        for j in 1:i-1
            A[:,i] .= view(A, :, i) - view(A, :, j) .* B[j,i]
        end
        A[:,i] .= view(A, :, i) ./ B[i,i]
    end
    return
end

function (ortho!::CholQR)(XBlocks::Blocks{Generalized}, sizeX = -1; update_AX=false, update_BX=false) where Generalized
    if sizeX == -1
        sizeX = size(XBlocks.block, 2)
    end
    X = XBlocks.block
    BX = XBlocks.B_block # Assumes it is premultiplied
    AX = XBlocks.A_block
    gram_view = view(ortho!.gramVBV, 1:sizeX, 1:sizeX)
    At_mul_B!(gram_view, view(X, :, 1:sizeX), view(BX, :, 1:sizeX))
    cholf = cholfact!(Hermitian(gram_view))
    R = cholf.factors
    A_rdiv_B!(X, UpperTriangular(R))
    update_AX && A_rdiv_B!(AX, UpperTriangular(R))
    Generalized && update_BX && A_rdiv_B!(BX, UpperTriangular(R))
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

function ortho_AB_mul_X!(blocks::Blocks, ortho!, A, B, bs=-1)
    # Finds BX
    bs == -1 ? B_mul_X!(blocks, B) : B_mul_X!(blocks, B, bs)
    # Orthonormalizes X and updates BX
    bs == -1 ? ortho!(blocks, update_BX=true) : ortho!(blocks, bs, update_BX=true)
    # Updates AX
    bs == -1 ? A_mul_X!(blocks, A) : A_mul_X!(blocks, A, bs)
    return 
end
function residuals!(rr)
    sizeX = size(rr.XBlocks.block, 2)
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
    return
end

function update_mask!(rr, residualTolerance)
    sizeX = size(rr.XBlocks.block, 2)
    # Update active vectors mask
    rr.activeMask .*= view(rr.residuals, 1:sizeX) .> residualTolerance
    rr.currentBlockSize[] = sum(rr.activeMask)
    return 
end

function update_active!(mask, bs::Int, blockPairs...)
    for (activeblock, block) in blockPairs
        activeblock[:, 1:bs] .= view(block, :, mask)
    end
    return
end

function precond_constr!(block, bs, precond!, constr!)
    precond!(view(block, 1:bs))
    # Constrain the active residual vectors to be B-orthogonal to Y
    constr!(view(block, 1:bs))
    return 
end
function block_grams_1x1!(rr)
    # Finds gram matrix X'AX
    XAX!(rr.gramABlock, rr.XBlocks)
    return
end
function block_grams_2x2!(rr, bs)
    sizeX = size(rr.XBlocks.block, 2)
    #XAX!(rr.gramABlock, rr.XBlocks)
    XAR!(rr.gramABlock, rr.XBlocks, rr.activeRBlocks, bs)
    RAR!(rr.gramABlock, rr.activeRBlocks, bs)
    XBR!(rr.gramBBlock, rr.XBlocks, rr.activeRBlocks, bs)        
    rr.gramABlock(rr.gramA, view(rr.λ, 1:sizeX), sizeX, bs, 0)
    rr.gramBBlock(rr.gramB, sizeX, bs, 0, true)

    return
end
function block_grams_3x3!(rr, bs)
    # Find R'AR, P'AP, X'AR, X'AP and R'AP
    sizeX = size(rr.XBlocks.block, 2)
    #XAX!(rr.gramABlock, rr.XBlocks)
    XAR!(rr.gramABlock, rr.XBlocks, rr.activeRBlocks, bs)
    XAP!(rr.gramABlock, rr.XBlocks, rr.activePBlocks, bs)
    RAR!(rr.gramABlock, rr.activeRBlocks, bs)
    RAP!(rr.gramABlock, rr.activeRBlocks, rr.activePBlocks, bs)
    PAP!(rr.gramABlock, rr.activePBlocks, bs)
    # Find X'BR, X'BP and P'BR
    XBR!(rr.gramBBlock, rr.XBlocks, rr.activeRBlocks, bs)
    XBP!(rr.gramBBlock, rr.XBlocks, rr.activePBlocks, bs)
    RBP!(rr.gramBBlock, rr.activeRBlocks, rr.activePBlocks, bs)    
    # Update the gram matrix [X R P]' A [X R P]
    rr.gramABlock(rr.gramA, view(rr.λ, 1:sizeX), sizeX, bs, bs)
    # Update the gram matrix [X R P]' B [X R P]
    rr.gramBBlock(rr.gramB, sizeX, bs, bs, true)

    return
end

function sub_problem!(rr, sizeX, bs1, bs2)
    subdim = sizeX+bs1+bs2
    gramAview = view(rr.gramA, 1:subdim, 1:subdim)
    if bs1 == 0
        eigf = eigfact!(Hermitian(rr.gramABlock.XAX))
    else
        gramBview = view(rr.gramB, 1:subdim, 1:subdim)
        eigf = eigfact!(Hermitian(gramAview), Hermitian(gramBview))
    end
    # Selects extremal eigenvalues and corresponding vectors
    selectperm!(view(rr.λperm, 1:subdim), eigf.values, 1:subdim, rev=rr.largest)
    rr.λ[1:sizeX] .= view(eigf.values, view(rr.λperm, 1:sizeX))
    rr.V[1:subdim, 1:sizeX] .= view(eigf.vectors, :, view(rr.λperm, 1:sizeX))

    return
end

function update_X_P!(rr::RayleighRitz{Generalized}, bs1, bs2) where Generalized
    sizeX = size(rr.XBlocks.block, 2)
    x_eigview = view(rr.V, 1:sizeX, 1:sizeX)
    r_eigview = view(rr.V, sizeX+1:sizeX+bs1, 1:sizeX)
    p_eigview = view(rr.V, sizeX+bs1+1:sizeX+bs1+bs2, 1:sizeX)
    r_blockview = view(rr.activeRBlocks.block, :, 1:bs1)
    ra_blockview = view(rr.activeRBlocks.A_block, :, 1:bs1)
    p_blockview = view(rr.activePBlocks.block, :, 1:bs2)
    pa_blockview = view(rr.activePBlocks.A_block, :, 1:bs2)
    if Generalized
        rb_blockview = view(rr.activeRBlocks.B_block, :, 1:bs1)
        pb_blockview = view(rr.activePBlocks.B_block, :, 1:bs2)
    end
    if bs1 > 0
        A_mul_B!(rr.PBlocks.block, r_blockview, r_eigview)
        A_mul_B!(rr.PBlocks.A_block, ra_blockview, r_eigview)
        if Generalized
            A_mul_B!(rr.PBlocks.B_block, rb_blockview, r_eigview)
        end
    end
    if bs2 > 0
        A_mul_B!(rr.tempXBlocks.block, p_blockview, p_eigview)
        A_mul_B!(rr.tempXBlocks.A_block, pa_blockview, p_eigview)
        if Generalized
            A_mul_B!(rr.tempXBlocks.B_block, pb_blockview, p_eigview)
        end
        rr.PBlocks.block .= rr.PBlocks.block .+ rr.tempXBlocks.block
        rr.PBlocks.A_block .= rr.PBlocks.A_block .+ rr.tempXBlocks.A_block
        if Generalized
            rr.PBlocks.B_block .= rr.PBlocks.B_block .+ rr.tempXBlocks.B_block
        end
    end
    block = rr.XBlocks.block
    tempblock = rr.tempXBlocks.block
    A_mul_B!(tempblock, block, x_eigview)
    block = rr.XBlocks.A_block
    tempblock = rr.tempXBlocks.A_block
    A_mul_B!(tempblock, block, x_eigview)
    if Generalized
        block = rr.XBlocks.B_block
        tempblock = rr.tempXBlocks.B_block
        A_mul_B!(tempblock, block, x_eigview)
    end
    if bs1 > 0
        rr.XBlocks.block .= rr.tempXBlocks.block .+ rr.PBlocks.block
        rr.XBlocks.A_block .= rr.tempXBlocks.A_block .+ rr.PBlocks.A_block
        if Generalized
            rr.XBlocks.B_block .= rr.tempXBlocks.B_block .+ rr.PBlocks.B_block
        end
    else
        rr.XBlocks.block .= rr.tempXBlocks.block
        rr.XBlocks.A_block .= rr.tempXBlocks.A_block
        if Generalized
            rr.XBlocks.B_block .= rr.tempXBlocks.B_block
        end
    end    
    return
end

function (rr::RayleighRitz{Generalized})(residualTolerance) where Generalized
    sizeX = size(rr.XBlocks.block, 2)
    iteration = rr.iteration[]
    if iteration == 1
        ortho_AB_mul_X!(rr.XBlocks, rr.ortho!, rr.A, rr.B)
        # Finds gram matrix X'AX
        block_grams_1x1!(rr)
        sub_problem!(rr, sizeX, 0, 0)
        # Updates Ritz vectors X and updates AX and BX accordingly
        update_X_P!(rr, 0, 0)
    elseif iteration == 2
        residuals!(rr)
        # Store history of norms
        push!(rr.residualNormsHistory, rr.residuals[1:sizeX])
        update_mask!(rr, residualTolerance)
        rr.currentBlockSize[] == 0 && return 
        bs = rr.currentBlockSize[]
        # Update active R blocks
        update_active!(rr.activeMask, bs, (rr.activeRBlocks.block, rr.RBlocks.block))
        # Precondition and constrain the active residual vectors
        precond_constr!(rr.activeRBlocks.block, bs, rr.precond!, rr.constr!)
        # Orthonormalizes R[:,1:bs] and finds AR[:,1:bs] and BR[:,1:bs]
        ortho_AB_mul_X!(rr.activeRBlocks, rr.ortho!, rr.A, rr.B, bs)
        # Find [X R] A [X R] and [X R]' B [X R]
        block_grams_2x2!(rr, bs)
        # Solve the Rayleigh-Ritz sub-problem
        sub_problem!(rr, sizeX, bs, 0)
        update_X_P!(rr, bs, 0)
    else
        residuals!(rr)
        # Store history of norms
        push!(rr.residualNormsHistory, rr.residuals[1:sizeX])
        update_mask!(rr, residualTolerance)
        rr.currentBlockSize[] == 0 && return
        # Update active blocks
        bs = rr.currentBlockSize[]
        # Update active R and P blocks
        update_active!(rr.activeMask, bs, (rr.activeRBlocks.block, rr.RBlocks.block), 
                                          (rr.activePBlocks.block, rr.PBlocks.block),
                                          (rr.activePBlocks.A_block, rr.PBlocks.A_block),
                                          (rr.activePBlocks.B_block, rr.PBlocks.B_block))
        # Precondition and constrain the active residual vectors
        precond_constr!(rr.activeRBlocks.block, bs, rr.precond!, rr.constr!)
        # Orthonormalizes R[:,1:bs] and finds AR[:,1:bs] and BR[:,1:bs]
        ortho_AB_mul_X!(rr.activeRBlocks, rr.ortho!, rr.A, rr.B, bs)
        # Orthonormalizes P and updates AP
        rr.ortho!(rr.activePBlocks, bs, update_AX=true, update_BX=true)
        # Update the gram matrix [X R P]' A [X R P] and [X R P]' B [X R P]
        block_grams_3x3!(rr, bs)
        # Solve the Rayleigh-Ritz sub-problem
        sub_problem!(rr, sizeX, bs, bs)
        # Updates Ritz vectors X and updates AX and BX accordingly
        # And updates P, AP and BP
        update_X_P!(rr, bs, bs)
    end

    return
end

function dense_solver(A, B, X, largest)
    warn("The problem size is small compared to the block size. Using dense eigensolver instead of LOBPCG.")
    # Define the closed range of indices of eigenvalues to return.
    n, sizeX = size(X)
    eigvals = largest ? (n - sizeX + 1, n) : (1, sizeX)
    A_dense = A*eye(n)
    if B isa Void
        return eig(Hermitian(A_dense))
    else
        B_dense = B*eye(n)
        return eig(Hermitian(A_dense), Hermitian(B_dense))
    end
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
        n < 5 * sizeX && return dense_solver(A, B, X, largest)
    else
        sizeY = size(Y, 2)
        (n - sizeY) < 5 * sizeX && throw("The dense eigensolver does not support constraints.")
    end
    sizeX > n && throw("X column dimension exceeds the row dimension")

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

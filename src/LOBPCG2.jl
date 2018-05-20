const DEBUG = false

macro debug(xs...)
    expr = Expr(:block)
    for x in xs
        push!(expr.args, :(@show $x))
    end
    push!(expr.args, :(println("-------------------------")))
    push!(expr.args, :(readline(STDIN)))
    esc(expr)
end

function pause()
    # Used only when verbosity level > 10.
    readline(STDIN)
end

"""Changes blockVectorV in place."""
function _applyConstraints(blockVectorV, factYBY, blockVectorBY, blockVectorY)
    gramYBV = At_mul_B(blockVectorBY, blockVectorV)
    tmp = factYBY\gramYBV
    blockVectorV .-= blockVectorY * tmp
end

function _b_orthonormalize(B, blockVectorV, blockVectorBV=nothing; retInvR=false)
    if blockVectorBV isa Void
        if !(B isa Void)
            blockVectorBV = B * blockVectorV
        else
            blockVectorBV = blockVectorV  # Shared data!!!
        end
    end
    @static if DEBUG
        println("Line = ", @__LINE__)
        @debug blockVectorV, blockVectorBV
    end

    gramVBV = At_mul_B(blockVectorV, blockVectorBV)
    @static if DEBUG
        println("Line = ", @__LINE__)
        @debug blockVectorV, blockVectorBV
    end
    gramVBV = (gramVBV + gramVBV')/2
    gramVBV = chol(gramVBV)
    gramVBV = inv(gramVBV)
    # gramVBV is now R^{-1}.
    if !(B isa Void)
        blockVectorV = blockVectorV * gramVBV
        blockVectorBV = blockVectorBV * gramVBV
    else
        blockVectorV = blockVectorV * gramVBV
        blockVectorBV = blockVectorV
    end

    if retInvR
        return blockVectorV, blockVectorBV, gramVBV
    else
        @static if DEBUG
            println("Line = ", @__LINE__)
            @debug blockVectorV, blockVectorBV
        end
        return blockVectorV, blockVectorBV
    end
end

"""Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)"""
function lobpcg(A, X,
            B=nothing, M=nothing, Y=nothing,
            tol=nothing, maxiter=200,
            largest=true, verbosityLevel=0,
            retLambdaHistory=false, retResidualNormsHistory=false)

    T = eltype(X)

    blockVectorX = X
    blockVectorY = Y
    residualTolerance = tol
    maxIterations = maxiter

    if !(blockVectorY isa Void)
        sizeY = size(blockVectorY, 2)
    else
        sizeY = 0
    end

    # Block size.
    if length(size(blockVectorX)) != 2
        throw("Expected 2D array for argument X")
    end

    n, sizeX = size(blockVectorX)
    if sizeX > n
        throw("X column dimension exceeds the row dimension")
    end

    if (n - sizeY) < (5 * sizeX)
        # warn('The problem size is small compared to the block size.' \
        #        ' Using dense eigensolver instead of LOBPCG.')

        if !(blockVectorY isa Void)
            throw("The dense eigensolver does not support constraints.")
        end

        # Define the closed range of indices of eigenvalues to return.
        if largest
            eigvals = (n - sizeX + 1, n)
        else
            eigvals = (1, sizeX)
        end

        A_dense = A * eye(n)
        if B isa Void
            B_dense = B
        else
            B_dense = B * eye(n)
        end

        return eig(A_dense, B_dense)
    end

    if residualTolerance isa Void
        residualTolerance = sqrt(1e-15)*n
    end
    maxIterations = min(n, maxIterations)

    if !(blockVectorY isa Void)
        if !(B isa Void)
            blockVectorBY = B * blockVectorY
        else
            blockVectorBY = blockVectorY
        end
        # gramYBY is a dense array.
        gramYBY = At_mul_B(blockVectorY, blockVectorBY)
        gramYBY = (gramYBY + gramYBY')/2
        try
            # gramYBY is a Cholesky factor from now on...
            gramYBY = cholfact(gramYBY)
        catch
            throw("cannot handle linearly dependent constraints")
        end
        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)
    end

    ##
    # B-orthonormalize X.
    blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)

    ##
    # Compute the initial Ritz vectors: solve the eigenproblem.
    blockVectorAX = A * blockVectorX
    gramXAX = At_mul_B(blockVectorX, blockVectorAX)

    _lambda, eigBlockVector = eig(gramXAX)

    ii = sortperm(_lambda)
    if largest
        reverse!(ii)
    end
    ii = ii[1:sizeX]

    _lambda = _lambda[ii]
    eigBlockVector = eigBlockVector[:,ii]

    blockVectorX = blockVectorX * eigBlockVector
    blockVectorAX = blockVectorAX * eigBlockVector
    if !(B isa Void)
        blockVectorBX = blockVectorBX * eigBlockVector
    end

    ##
    # Active index set.
    activeMask = ones(Bool, (sizeX,))

    lambdaHistory = [_lambda]
    residualNormsHistory = []

    previousBlockSize = sizeX
    ident = eye(T, sizeX)
    ident0 = eye(T, sizeX)

    ##
    # Main iteration loop.

    blockVectorP = nothing  # set during iteration
    blockVectorAP = nothing
    blockVectorBP = nothing

    for iterationNumber in 1:maxIterations
        if verbosityLevel > 0
            print("iteration $iterationNumber")
        end

        aux = blockVectorBX * diagm(_lambda)
        blockVectorR = blockVectorAX - aux

        aux = reshape(mapreducedim((x)->(conj(x)*x), +, blockVectorR, 1), (sizeX,))
        residualNorms = sqrt.(aux)

        push!(residualNormsHistory, residualNorms)

        for i in 1:sizeX
            activeMask[i] = activeMask[i] && residualNorms[i] > residualTolerance
        end
        @static if DEBUG
            println("Line = ", @__LINE__)
            @debug activeMask
        end

        if verbosityLevel > 2
            println(activeMask)
        end

        currentBlockSize = sum(activeMask)
        if currentBlockSize != previousBlockSize
            previousBlockSize = currentBlockSize
            ident = eye(T, currentBlockSize)
        end
        
        if currentBlockSize == 0
            break
        end

        if verbosityLevel > 0
            println("current block size:", currentBlockSize)
            println("eigenvalue:", _lambda)
            println("residual norms:", residualNorms)
        end
        if verbosityLevel > 10
            println(eigBlockVector)
        end

        activeBlockVectorR = blockVectorR[:,activeMask]

        if iterationNumber > 1
            @static if DEBUG
                println("Line = ", @__LINE__)
                @debug blockVectorP, blockVectorBP
            end

            activeBlockVectorP = blockVectorP[:,activeMask]
            activeBlockVectorAP = blockVectorAP[:,activeMask]
            activeBlockVectorBP = blockVectorBP[:,activeMask]

            @static if DEBUG
                println("Line = ", @__LINE__)
                @debug activeBlockVectorP, activeBlockVectorBP
            end
        end

        if !(M isa Void)
            # Apply preconditioner T to the active residuals.
            activeBlockVectorR = M * activeBlockVectorR
        end

        ##
        # Apply constraints to the preconditioned residuals.
        if !(blockVectorY isa Void)
            @static if DEBUG
                println("Line = ", @__LINE__)
                @debug blockVectorY, blockVectorBY
            end
            _applyConstraints(activeBlockVectorR,
                              gramYBY, blockVectorBY, blockVectorY)
        end

        ##
        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux

        @static if DEBUG
            println("Line = ", @__LINE__)
            @debug activeBlockVectorR, activeBlockVectorBR
        end

        activeBlockVectorAR = A * activeBlockVectorR

        if iterationNumber > 1
            aux = _b_orthonormalize(B, activeBlockVectorP,
                                    activeBlockVectorBP, retInvR=true)
            activeBlockVectorP, activeBlockVectorBP, invR = aux
            @static if DEBUG
                println("Line = ", @__LINE__)
                @debug activeBlockVectorP, activeBlockVectorBP
            end
            activeBlockVectorAP = activeBlockVectorAP * invR
        end

        ##
        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:

        xaw = At_mul_B(blockVectorX, activeBlockVectorAR)
        waw = At_mul_B(activeBlockVectorR, activeBlockVectorAR)
        xbw = At_mul_B(blockVectorX, activeBlockVectorBR)

        if iterationNumber > 1
            xap = At_mul_B(blockVectorX, activeBlockVectorAP)
            wap = At_mul_B(activeBlockVectorR, activeBlockVectorAP)
            pap = At_mul_B(activeBlockVectorP, activeBlockVectorAP)
            xbp = At_mul_B(blockVectorX, activeBlockVectorBP)
            wbp = At_mul_B(activeBlockVectorR, activeBlockVectorBP)

            gramA = [diagm(_lambda) xaw xap
                    xaw' waw wap
                    xap' wap' pap]

            gramB = [ident0 xbw xbp
                    xbw' ident wbp
                    xbp' wbp' ident]
        else
            gramA = [diagm(_lambda) xaw
                    xaw' waw]
            gramB = [ident0 xbw
                    xbw' ident]
        end

        gramA = (gramA + gramA')/2
        gramB = (gramB + gramB')/2

        @assert ishermitian(gramA)
        @assert ishermitian(gramB)

        if verbosityLevel > 10
            save("gramA.jld2", gramA)
            save("gramB.jld2", gramB)
        end

        # Solve the generalized eigenvalue problem.
        _lambda, eigBlockVector = eig(gramA, gramB)

        ii = sortperm(_lambda)
        if largest
            reverse!(ii)
        end
        ii = ii[1:sizeX]

        if verbosityLevel > 10
            @show ii
        end

        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:,ii]

        push!(lambdaHistory, _lambda)

        if verbosityLevel > 10
            println("lambda:", _lambda)
        end

        if verbosityLevel > 10
            println(eigBlockVector)
            pause()
        end

        # Compute Ritz vectors.
        if iterationNumber > 1
            eigBlockVectorX = eigBlockVector[1:sizeX, :]
            eigBlockVectorR = eigBlockVector[sizeX+1:sizeX+currentBlockSize, :]
            eigBlockVectorP = eigBlockVector[sizeX+currentBlockSize+1:end, :]

            pp = activeBlockVectorR * eigBlockVectorR
            pp += activeBlockVectorP * eigBlockVectorP

            app = activeBlockVectorAR * eigBlockVectorR
            app += activeBlockVectorAP * eigBlockVectorP

            bpp = activeBlockVectorBR * eigBlockVectorR
            bpp += activeBlockVectorBP * eigBlockVectorP

            @static if DEBUG
                println("Line = ", @__LINE__)
                @debug pp, bpp
            end
        else
            eigBlockVectorX = eigBlockVector[1:sizeX, :]
            eigBlockVectorR = eigBlockVector[sizeX+1:end, :]
            pp = activeBlockVectorR * eigBlockVectorR
            app = activeBlockVectorAR * eigBlockVectorR
            bpp = activeBlockVectorBR * eigBlockVectorR

            @static if DEBUG
                println("Line = ", @__LINE__)
                @debug pp, bpp
            end
        end

        if verbosityLevel > 10
            println(pp)
            println(app)
            println(bpp)
            pause()
        end

        @static if DEBUG
            println("Line = ", @__LINE__)
            @debug blockVectorX, blockVectorBX
        end
        blockVectorX = blockVectorX * eigBlockVectorX + pp
        blockVectorAX = blockVectorAX * eigBlockVectorX + app
        blockVectorBX = blockVectorBX * eigBlockVectorX + bpp

        @static if DEBUG
            println("Line = ", @__LINE__)
            @debug blockVectorX, blockVectorBX
        end
        blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp
        @static if DEBUG
            println("Line = ", @__LINE__)
            @debug blockVectorP, blockVectorBP
        end
    end
    
    aux = blockVectorBX * diagm(_lambda)
    blockVectorR = blockVectorAX - aux

    aux = reshape(mapreducedim((x)->(conj(x)*x), +, blockVectorR, 1), (sizeX,))
    residualNorms = sqrt.(aux)

    if verbosityLevel > 0
        println("final eigenvalue:", _lambda)
        println("final residual norms:", residualNorms)
    end

    if retLambdaHistory
        if retResidualNormsHistory
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else
            return _lambda, blockVectorX, lambdaHistory
        end
    else
        if retResidualNormsHistory
            return _lambda, blockVectorX, residualNormsHistory
        else
            return _lambda, blockVectorX
        end
    end
end

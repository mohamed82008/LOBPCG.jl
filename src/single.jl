function locg(A, ::UniformScaling, x::Vector{T}, ::Type{Val{sense}}=Val{:Min}, tol=T(1)/10^6, P! = identity, maxiter = 500) where {T, sense}
    buffers = LOCGBuffersSimple{Vector{T}, Matrix{T}}(x, Matrix{T}(length(x), 3))
    locg(A, buffers, Val{sense}, tol, P!, maxiter)
end

function locg(A, B, x::Vector{T}, ::Type{Val{sense}}=Val{:Min}, tol=T(1)/10^6, P! = identity, maxiter = 500) where {T, sense}
    buffers = LOCGBuffersGeneral{Vector{T}, Matrix{T}}(x, Matrix{T}(length(x), 3))
    locg(A, B, buffers, Val{sense}, tol, P!, maxiter)
end

function locg(A, x::Vector{T}, ::Type{Val{sense}}=Val{:Min}, tol=eltype(x)(1)/10^6, P! = identity, maxiter = 500) where {T, sense}
    buffers = LOCGBuffersSimple{Vector{T}, Matrix{T}}(x, Matrix{T}(length(x), 3))
    locg(A, buffers, Val{sense}, tol, P!, maxiter)
end

function locg(A, buffers::LOCGBuffersSimple{Tx,TQ}, ::Type{Val{sense}}=Val{:Min}, tol=eltype(buffers.x)(1)/10^6, P! = identity, maxiter = 500) where {Tx, TQ, sense}
    T = eltype(buffers.x)

    x, Ax, x_prev, Ax_prev, r, Ar, c, X, Q = buffers.x, buffers.Ax, buffers.x_prev, buffers.Ax_prev, buffers.r, buffers.Ar, buffers.c, buffers.X, buffers.Q

    # Normalize x_prev
    x_norm = sqrt(dot(x_prev, x_prev))
    @inbounds x_prev ./= x_norm
    A_mul_B!(Ax_prev, A, x_prev)

    # Normalize x
    x_norm = sqrt(dot(x, x))
    @inbounds x ./= x_norm
    A_mul_B!(Ax, A, x)

    # Find Rayleight quotient
    lambda::T = dot(Ax, x)

    # Find residual
    @inbounds r .= Ax .- lambda .* x

    # Find residual norm
    r_norm = sqrt(dot(r, r))

    iter = 0
    while r_norm > tol && iter < maxiter
        iter += 1
        # Precondition new search direction
        if !(P! isa typeof(identity))
            P!(r)
            r_norm = sqrt(dot(r, r))
        end
        # Normalize residual
        @inbounds r ./= r_norm
        A_mul_B!(Ar, A, r)

        # Enforce independence of search directions
        G::SMatrix{3,3,T,9} = gram_matrix_I(x_prev, x, r)
        if singular(G)
            # Overwrite search directions
            overwrite_qr!(X, Q, x_prev, x, r)
            tmpX = X
            X = Q
            Q = tmpX

            # Restricted matrix A
            R_A = restricted_matrix!(Q, A, X)
            
            # Restricted eigenvalue problem
            vals, vects = eig(Symmetric(R_A))

            # New Rayleigh quotient
            if sense === :Min
                lambda, ind = findmin(vals)
            else
                lambda, ind = findmax(vals)
            end

            # Update solution/search directions
            @inbounds x_prev .= x
            @inbounds Ax_prev .= Ax
            v1 = vects[1,ind]
            v2 = vects[2,ind]
            v3 = vects[3,ind]
            @inbounds @simd for i in 1:size(X,1)
                x[i] = X[i,1] * v1 + X[i,2] * v2 + X[i,3] * v3
            end
            A_mul_B!(Ax, A, x)

        else
            # Restricted matrix A
            R_A = restricted_matrix(x_prev, x, r, Ax_prev, Ax, Ar)

            # Restricted generalized eigenvalue problem
            vals::SVector{3,T}, vects::SMatrix{3,3,T,9} = general_eig(R_A, G)

            # New Rayleigh quotient
            if sense === :Min
                lambda, ind = findmin(vals)
            else
                lambda, ind = findmax(vals)
            end

            # Update solution/search directions
            v1 = vects[1,ind]
            v2 = vects[2,ind]
            v3 = vects[3,ind]
            @inbounds @simd for i in 1:length(x)
                tmp1 = x[i]
                x[i] = v1 * x_prev[i] + v2 * x[i] + v3 * r[i]
                x_prev[i] = tmp1

                tmp2 = Ax[i]
                Ax[i] = v1 * Ax_prev[i] + v2 * Ax[i] + v3 * Ar[i]
                Ax_prev[i] = tmp2
            end
        end
        
        # Find the residual and its norm
        @inbounds r .= Ax .- lambda .* x
        r_norm = sqrt(dot(r, r))
    end
    return lambda::T, x::Tx
end

function locg(A, B, buffers::LOCGBuffersGeneral{Tx,TQ}, ::Type{Val{sense}}=Val{:Min}, tol=eltype(x)(1)/10^6, P! = identity, maxiter = 500) where {Tx, TQ, sense}
    T = eltype(buffers.x)

    x, Ax, Bx, x_prev, Ax_prev, Bx_prev, r, Ar, Br, c, X, Q = buffers.x, buffers.Ax, buffers.Bx, buffers.x_prev, buffers.Ax_prev, buffers.Bx_prev, buffers.r, buffers.Ar, buffers.Br, buffers.c, buffers.X, buffers.Q
    
    # Normalize x_prev
    A_mul_B!(Bx_prev, B, x_prev)
    x_norm = sqrt(dot(Bx_prev, x_prev))
    @inbounds x_prev ./= x_norm
    @inbounds Bx_prev ./= x_norm
    A_mul_B!(Ax_prev, A, x_prev)

    # Normalize x
    A_mul_B!(Bx, B, x)
    x_norm = sqrt(dot(Bx, x))
    @inbounds x ./= x_norm
    Bx ./= x_norm
    A_mul_B!(Ax, A, x)

    # Find Rayleight quotient
    lambda::T = dot(Ax, x)

    # Find residual
    @inbounds r .= Ax .- lambda .* Bx

    # Find residual norm and check convergence
    r_norm = sqrt(dot(r, r))

    iter = 0
    while r_norm > tol && iter < maxiter
        iter += 1
        # Preconditioner new search direction
        if !(P! isa typeof(identity))
            P!(r)
        end
        # Normalize residual
        A_mul_B!(Br, B, r)
        r_norm = sqrt(dot(Br, r))
        @inbounds r ./= r_norm
        @inbounds Br ./= r_norm
        A_mul_B!(Ar, A, r)

        # Enforce independence of search directions
        R_B = restricted_matrix_I(x_prev, x, r, Bx_prev, Bx, Br)
        if singular(R_B)
            # Overwrite search directions
            overwrite_qr!(X, Q, x_prev, x, r)
            tmpX = X
            X = Q
            Q = tmpX

            R_A = restricted_matrix!(Q, A, X)
            R_B = restricted_matrix!(Q, B, X)
            vals, vects = general_eig(R_A, R_B)

            # New Rayleigh quotient
            if sense === :Min
                lambda, ind = findmin(vals)
            else
                lambda, ind = findmax(vals)
            end

            # Update solution/search directions
            @inbounds begin 
                x_prev .= x
                Ax_prev .= Ax
                Bx_prev .= Bx
                v1 = vects[1,ind]
                v2 = vects[2,ind]
                v3 = vects[3,ind]
            end

            @inbounds @simd for i in 1:size(X,1)
                x[i] = X[i,1] * v1 + X[i,2] * v2 + X[i,3] * v3
            end

            A_mul_B!(Ax, A, x)
            A_mul_B!(Bx, B, x)
        else
            R_A = restricted_matrix(x_prev, x, r, Ax_prev, Ax, Ar)
            vals, vects = general_eig(R_A, R_B)

            # New Rayleigh quotient
            if sense === :Min
                lambda, ind = findmin(vals)
            else
                lambda, ind = findmax(vals)
            end

            # Update solution/search directions
            @inbounds begin
                v1 = vects[1,ind]
                v2 = vects[2,ind]
                v3 = vects[3,ind]
            end
            @inbounds @simd for i in 1:length(x)
                tmp1 = x[i]
                x[i] = v1 * x_prev[i] + v2 * x[i] + v3 * r[i]
                x_prev[i] = tmp1

                tmp2 = Ax[i]
                Ax[i] = v1 * Ax_prev[i] + v2 * Ax[i] + v3 * Ar[i]
                Ax_prev[i] = tmp2

                tmp3 = Bx[i]
                Bx[i] = v1 * Bx_prev[i] + v2 * Bx[i] + v3 * Br[i]
                Bx_prev[i] = tmp3
            end
        end
        
        x_norm = sqrt(dot(Bx, x))
        @inbounds x ./= x_norm
        @inbounds Ax ./= x_norm
        @inbounds Bx ./= x_norm
        
        # Find residual
        @inbounds r .= Ax .- lambda .* Bx
        r_norm = sqrt(dot(r, r))
    end
    return lambda::T, x::Tx
end

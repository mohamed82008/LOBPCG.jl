function singular(G)
    T = eltype(G)
    return det(G) <= eps(T)
end

function gram_matrix(X)
    dp12::eltype(X) = innerproduct(X,X,1,2)
    dp13::eltype(X) = innerproduct(X,X,1,3)
    dp23::eltype(X) = innerproduct(X,X,2,3)
    o::eltype(X) = one(eltype(X))
    return SMatrix{3,3,eltype(X),9}(
        (o, dp12, dp13,
        dp12, o, dp23,
        dp13, dp23, o)
    )
end

function gram_matrix_I(x1, x2, x3)
    T = eltype(x1)
    dp12::T = dot(x1,x2)
    dp13::T = dot(x1,x3)
    dp23::T = dot(x2,x3)
    o = one(T)
    return SMatrix{3,3,T,9}(
        (o, dp12, dp13,
        dp12, o, dp23,
        dp13, dp23, o)
    )
end

function innerproduct(X1, X2, i, j)
    p = zero(eltype(X1))
    for k in 1:size(X1,1)
        p += X1[k,i]*X2[k,j]
    end
    p
end

function matrix_inner_product(X1, X2)
    dp11 = innerproduct(X1,X2,1,1)
    dp12 = innerproduct(X1,X2,1,2)
    dp13 = innerproduct(X1,X2,1,3)
    dp22 = innerproduct(X1,X2,2,2)
    dp23 = innerproduct(X1,X2,2,3)
    dp33 = innerproduct(X1,X2,3,3)
    return SMatrix{3,3,eltype(X1),9}((
        dp11, dp12, dp13,
        dp12, dp22, dp23,
        dp13, dp23, dp33
    ))
end

function restricted_matrix!(C, A, X)
    A_mul_B!(C, A, X)
    R::SMatrix{3,3,Float64,9} = matrix_inner_product(X, C)
    return R
end

function restricted_matrix(x_prev, x, r, Ax_prev, Ax, Ar)
    T = eltype(x)
    dp11 = dot(Ax_prev, x_prev)
    dp12 = dot(Ax_prev, x)
    dp13 = dot(Ax_prev, r)
    dp22 = dot(Ax, x)
    dp23 = dot(Ax, r)
    dp33 = dot(Ar, r)
    return SMatrix{3,3,T,9}((
        dp11, dp12, dp13,
        dp12, dp22, dp23,
        dp13, dp23, dp33
    ))
end

function restricted_matrix_I(x_prev, x, r, Ax_prev, Ax, Ar)
    T = eltype(x)
    o = one(T)
    dp12 = dot(Ax_prev, x)
    dp13 = dot(Ax_prev, r)
    dp23 = dot(Ax, r)
    return SMatrix{3,3,T,9}((
        o, dp12, dp13,
        dp12, o, dp23,
        dp13, dp23, o
    ))
end

function general_eig(A, B)
    C = chol(Symmetric(B))
    Ct = C'
    C_inv = inv(C)
    vals, vects = eig(Symmetric(C_inv' * A * C_inv))
    return vals, C_inv*vects
end

function overwrite_qr!(X, Q, x_prev, x, r)            
    T = eltype(X)
    @inbounds X[:,1] .= x_prev
    @inbounds X[:,2] .= x
    @inbounds X[:,3] .= r

    # QR factorization
    qrf = qrfact!(X)
    @inbounds Q .= zero(T)
    @inbounds begin
        Q[1,1] = one(T)                    
        Q[2,2] = one(T)                    
        Q[3,3] = one(T)                    
    end
    A_mul_B!(qrf[:Q], Q)
    return
end

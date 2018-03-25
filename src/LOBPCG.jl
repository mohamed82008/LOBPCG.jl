module LOBPCG

export locg

function rq!(c, A, x) # Assumes x is normalized with respect to B
    A_mul_B!(c, A, x)
    return dot(c,x)
end

function residual!(r, c, A, B, x, lambda)
    A_mul_B!(r, A, x)
    A_mul_B!(c, B, x)
    r .-= lambda .* c
    return r
end
function residual!(r, c, A, ::UniformScaling, x, lambda)
    A_mul_B!(r, A, x)
    r .-= lambda .* x
    return r
end

innerproduct(X, i, j) = mapreduce((x)->(X[x,i]*X[x,j]), +, 1:size(X,1))
function gram_matrix!(G, X)
    for (i,j) in Base.Iterators.product(indices(G)...)
        G[i,j] = innerproduct(X,i,j)
    end
    return G
end
function gram_matrix!(G, x...)
    for (i,j) in Base.Iterators.product(indices(G)...)
        G[i,j] = dot(x[i],x[j])
    end
    return G
end

innerproduct(X1, X2, i, j) = mapreduce((x)->(X1[x,i]*X2[x,j]), +, 1:size(X1,1))
function matrix_inner_product!(G, X1, X2)
    for (i,j) in Base.Iterators.product(indices(G)...)
        G[i,j] = innerproduct(X1,X2,i,j)
    end
    return G
end
function restricted_matrix!(R, C, A, X)
    A_mul_B!(C, A, X)
    matrix_inner_product!(R, X, C)
    return R
end
function restricted_matrix!(R, C, ::UniformScaling, X)
    matrix_inner_product!(R, X, X)
    return R
end

function norm(x, c, A)
    A_mul_B!(c, A, x)
    return sqrt(dot(x,c))
end
function norm(x, c, ::UniformScaling)
    return sqrt(dot(x,x))
end
function normalize!(x, c, A)
    x ./= norm(x, c, A)
end

function locg(A, B, x, ::Type{Val{sense}}=Val{:Min}; tol=eltype(x)(1)/10^6) where sense
    T = eltype(x)

    c = zeros(T, length(x)) # Matrix-vector multiplication buffer
    r = zeros(T, length(x)) # Residual
    R_A = zeros(T,3,3) # Restricted matrix A
    R_B = zeros(T,3,3) # Restricted matrix B
    C = zeros(T,length(x),3) # Matrix-matrix multiplication buffer
    G = zeros(T,3,3) # Gram matrix
    X = zeros(T,length(x),3) # Matrix of search directions

    # Initialize x
    if all((x)->(x == zero(T)), x)
        x = rand(T, length(x))
    end
    normalize!(x, c, B)
    # Find Rayleight quotient
    lambda = rq!(c, A, x)

    # Find residual
    residual!(r, c, A, B, x, lambda)
    r_norm = norm(r, c, B)
    if r_norm < tol
        return x, lambda
    end
    r ./= r_norm

    # Generate a random search direction
    x_prev = rand(T, length(x))
    normalize!(x_prev, c, B)

    while r_norm > tol
        # Overwrite search directions
        X[:,1] .= x_prev
        X[:,2] .= x
        X[:,3] .= r

        # Enforce independence of search directions
        gram_matrix!(G, X) # Can compute only half
        if !isposdef(Symmetric(G))
            X = qr(X)[1] # Should replace with an inplace version
        end

        # Compute restricted problem eigenvalues and vectors
        restricted_matrix!(R_A, C, A, X)
        restricted_matrix!(R_B, C, B, X)
        vals, vects = eig(R_A, R_B) # Should replace with inplace version or use StaticArrays
        if sense === :Min
            lambda, ind = findmin(vals) # New Rayleigh quotient
            x_r = vects[:,ind] # Make more efficient
        else
            lambda, ind = findmax(vals) # New Rayleigh quotient
            x_r = vects[:,ind] # Make more efficient
        end
        # Update solution/search directions
        x_prev .= x
        A_mul_B!(x, X, x_r)
        normalize!(x, c, B)

        # Find residual
        residual!(r, c, A, B, x, lambda)
        r_norm = norm(r, c, B)
        if r_norm < tol
            return lambda, x
        end
        r ./= r_norm
    end
end

end # module

struct LOCGBuffersSimple{Tx,TQ}
    x::Tx
    Ax::Tx
    x_prev::Tx
    Ax_prev::Tx
    r::Tx
    Ar::Tx
    c::Tx
    X::TQ
    Q::TQ
end
function LOCGBuffersSimple(x::Vector{T}) where T
    Q = Matrix{T}(length(x), 3)
    if all((x)->(x == zero(T)), x)
        x = rand(T, length(x))
    else
        x = copy(x)
    end
    LOCGBuffersSimple{Vector{T}, Matrix{T}}(x, similar(x), rand(T, length(x)), similar(x), similar(x), similar(x), similar(x), Q, similar(Q))
end
function LOCGBuffersSimple{Tx,TQ}(x::Tx, Q::TQ) where {Tx<:AbstractVector, TQ<:AbstractMatrix}
    length(x) == size(Q,1) || throw("Length of x is not equal to size(Q,2).") 
    size(Q,2) == 3 || throw("size(Q,2) is not equal to 3.")
    if all((x)->(x == zero(eltype(x))), x)
        x = rand(eltype(x), length(x))
    else
        x = copy(x)
    end
    LOCGBuffersSimple{Tx,TQ}(x, similar(x), rand(eltype(x), length(x)), similar(x), similar(x), similar(x), similar(x), Q, similar(Q))
end

struct LOCGBuffersGeneral{Tx,TQ}
    x::Tx
    Ax::Tx
    Bx::Tx
    x_prev::Tx
    Ax_prev::Tx
    Bx_prev::Tx
    r::Tx
    Ar::Tx
    Br::Tx
    c::Tx
    X::TQ
    Q::TQ
end
function LOCGBuffersGeneral(x::Vector{T}) where T
    Q = Matrix{T}(length(x), 3)
    if all((x)->(x == zero(T)), x)
        x = rand(T, length(x))
    else
        x = copy(x)
    end
    LOCGBuffersGeneral{Vector{T}, Matrix{T}}(x, similar(x), similar(x), rand(T, length(x)), similar(x), similar(x), similar(x), similar(x), similar(x), similar(x), Q, similar(Q))
end
function LOCGBuffersGeneral{Tx,TQ}(x::Tx, Q::TQ) where {Tx<:AbstractVector, TQ<:AbstractMatrix}
    length(x) == size(Q,1) || throw("Length of x is not equal to size(Q,2).") 
    size(Q,2) == 3 || throw("size(Q,2) is not equal to 3.")
    if all((x)->(x == zero(eltype(x))), x)
        x = rand(eltype(x), length(x))
    else
        x = copy(x)
    end
    LOCGBuffersGeneral{Tx,TQ}(x, similar(x), similar(x), rand(eltype(x), length(x)), similar(x), similar(x), similar(x), similar(x), similar(x), similar(x), Q, similar(Q))
end


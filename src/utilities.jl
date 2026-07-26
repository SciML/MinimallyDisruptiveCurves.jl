"""
    sparse_init_dir(
        hessian; orthogonal_to = [], λ = 1.0, start = nothing,
        trim_level = 1.0e-5, max_iter = 2000, tol = 1.0e-6
    )

Compute a sparse, unit-norm initial direction associated with a low-curvature
direction of a symmetric Hessian.

# Arguments
- `hessian`: square real matrix. It should be symmetric; nonsymmetric input
  does not have the intended curvature interpretation.

# Keyword Arguments
- `orthogonal_to`: vectors the result is projected away from before
  normalization.
- `λ::Real=1.0`: relative soft-threshold strength, scaled by the largest
  eigenvalue.
- `start`: optional initial vector; otherwise the smallest-eigenvalue vector
  is used.
- `trim_level::Real=1e-5`: entries with smaller absolute value are zeroed.
- `max_iter::Integer=2000`: maximum proximal-gradient iterations.
- `tol::Real=1e-6`: convergence threshold on successive iterates.

# Returns
`(direction, curvature)`, where `direction` is unit norm unless thresholding
collapses it to zero, and `curvature == dot(direction, hessian, direction)`.

# Example
```julia
direction, curvature = sparse_init_dir([1.0 0.0; 0.0 4.0])
```
"""
function sparse_init_dir(hessian; orthogonal_to = Vector{Vector{Float64}}(), λ = 1.0, start = nothing, trim_level = 1.0e-5, max_iter = 2000, tol = 1.0e-6)
    n = size(hessian, 1)

    # 1. Initialisation & Scale Estimation
    E = eigen(hessian)
    H_scale = E.values[end]
    effective_λ = λ * H_scale
    t = 1.0 / (2.0 * H_scale)

    if isnothing(start)
        x = copy(E.vectors[:, 1])
    else
        x = convert(Vector{eltype(hessian)}, copy(start))
    end

    # Pre-allocate cache vectors to achieve 0 allocations in the loop
    x_old = copy(x)
    grad_smooth = similar(x)
    diff_cache = similar(x)

    # Ensure starting vector is properly orthogonalised from the get-go
    if !isempty(orthogonal_to)
        for el in orthogonal_to
            x .-= dot(x, el) .* el
        end
        normalize!(x)
    end

    for iter in 1:max_iter
        copyto!(x_old, x)

        # Step A: In-place Gradient Calculation
        # grad = 2 * H * x
        mul!(grad_smooth, hessian, x)

        # Step B: Gradient Descent + Proximal Operator combined
        @inbounds for i in 1:n
            # Compute the forward gradient step explicitly per element
            xi = x[i] - t * 2.0 * grad_smooth[i]
            # Apply soft-thresholding
            x[i] = sign(xi) * max(0.0, abs(xi) - t * effective_λ)
        end

        # Step C: Project onto the orthogonal subspace BEFORE normalisation
        if !isempty(orthogonal_to)
            for el in orthogonal_to
                dot_prod = dot(x, el)
                @. x -= dot_prod * el  # Fusion prevents allocation
            end
        end

        # Step D: Project back onto unit sphere
        nx = norm(x)
        if nx > 1.0e-8
            x ./= nx
        else
            # If λ crushed the vector, we attempt a reset to a valid subspace element
            # instead of returning a hard failure zero-vector.
            @warn "λ parameter is too aggressive; vector collapsed to zero. Forcing reset."
            x .= 0.0
            return x, 0.0
        end

        # Convergence Check
        @. diff_cache = x - x_old
        if norm(diff_cache) < tol
            break
        end
    end

    # Hard-threshold anything beneath trim level
    @. x = ifelse(abs(x) < trim_level, 0.0, x)

    # Final re-normalisation
    nx = norm(x)
    if nx > 1.0e-8
        x ./= nx
    else
        x .= 0.0
    end

    val = dot(x, hessian, x) # Allocation-free x' * H * x
    return x, val
end

"""
    sparse_eigenbasis(
        hessian, num_vectors; λ = 1.0, trim_level = 1.0e-5,
        max_iter = 2000, tol = 1.0e-6
    )

Build up to `num_vectors` sparse, mutually projected initial MDC directions.

# Arguments
- `hessian`: square real Hessian matrix.
- `num_vectors::Integer`: requested number of directions, no larger than the
  Hessian dimension.

# Keyword Arguments
- `λ`, `trim_level`, `max_iter`, `tol`: forwarded to [`sparse_init_dir`](@ref).

# Returns
`(basis, values)`, where `basis` contains each accepted sparse direction and
`values` contains its Rayleigh quotient. The result can contain fewer than
`num_vectors` entries if sparsification collapses a later direction.

# Example
```julia
basis, values = sparse_eigenbasis([1.0 0.0; 0.0 4.0], 2)
```
"""
function sparse_eigenbasis(hessian, num_vectors::Int; λ = 1.0, trim_level = 1.0e-5, max_iter = 2000, tol = 1.0e-6)
    n = size(hessian, 1)
    if num_vectors > n
        error("num_vectors ($num_vectors) cannot exceed the dimension of the Hessian ($n).")
    end

    basis = Vector{Vector{eltype(hessian)}}()
    values = eltype(hessian)[]

    # Cache the full eigendecomposition once outside the loop so we can pass
    # good initial guesses (the standard eigenvectors) to each sequential step
    E = eigen(hessian)

    for i in 1:num_vectors
        start_guess = copy(E.vectors[:, i])

        # Pass the accumulated basis vectors directly.
        # It starts as an empty vector, which maintains type stability!
        orthogonal_list = basis

        # Find the next sparse direction
        v_sparse, val = sparse_init_dir(
            hessian;
            orthogonal_to = orthogonal_list,
            λ = λ,
            start = start_guess,
            trim_level = trim_level,
            max_iter = max_iter,
            tol = tol
        )

        # If the parameter lambda is too aggressive, it might kill off higher dimensions.
        if all(v_sparse .== 0.0)
            @warn "Algorithm collapsed to a zero vector at basis element $i. Stopping early."
            break
        end

        push!(basis, v_sparse)
        push!(values, val)
    end

    return basis, values
end

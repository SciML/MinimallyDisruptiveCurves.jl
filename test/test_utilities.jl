using SafeTestsets

@safetestset "Sparse Initialization Utilities" begin
    using MinimallyDisruptiveCurves
    using MinimallyDisruptiveCurves: sparse_init_dir, sparse_eigenbasis
    using Test
    using LinearAlgebra

    @testset "sparse_init_dir" begin
        # Simple 4x4 diagonal Hessian
        hessian = diagm([10.0, 1.0, 0.1, 5.0])
        
        # Find the first sparse direction (should target the smallest eigenvalue 0.1)
        v, val = sparse_init_dir(hessian; λ=0.01, max_iter=500)
        
        @test norm(v) ≈ 1.0 atol=1e-6  # Must be normalized
        @test val ≈ 0.1 rtol=1e-2       # Rayleigh quotient should approach the smallest eigenvalue
        
        # Test orthogonality enforcement
        v1, _ = sparse_init_dir(hessian; λ=0.01)
        # Pass v1 as the orthogonal constraint
        v2, _ = sparse_init_dir(hessian; orthogonal_to=[v1], λ=0.01)
        
        @test norm(v1) ≈ 1.0 atol=1e-6
        @test abs(dot(v1, v2)) < 1e-5   # Must be orthogonal to v1
    end

    @testset "sparse_eigenbasis" begin
        # 4x4 Hessian with distinct eigenvalues
        hessian = diagm([10.0, 1.0, 0.1, 5.0])
        
        num_vecs = 3
        basis, values = sparse_eigenbasis(hessian, num_vecs; λ=0.01, max_iter=500)
        
        # Should return exactly the requested number of vectors (if λ isn't too aggressive)
        @test length(basis) == num_vecs
        @test length(values) == num_vecs
        
        # All basis vectors should be individually normalized
        for v in basis
            @test norm(v) ≈ 1.0 atol=1e-6
        end
        
        # All basis vectors should be mutually orthogonal
        for i in 1:num_vecs
            for j in (i+1):num_vecs
                @test abs(dot(basis[i], basis[j])) < 1e-5
            end
        end
        
        # Values (Rayleigh quotients) should be close to the true smallest eigenvalues
        true_eigenvalues = sort([10.0, 1.0, 0.1, 5.0])[1:num_vecs]
        @test values ≈ true_eigenvalues rtol=0.1
    end
end

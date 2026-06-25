using SciMLTesting
using JET
using MinimallyDisruptiveCurves
using Test

run_qa(
    MinimallyDisruptiveCurves;
    explicit_imports = true,
    ei_kwargs = (;
        # Base.front / Base.tail are non-public Base tuple helpers accessed via
        # qualified access in src/transforms.jl and src/MDCProblem.jl.
        all_qualified_accesses_are_public = (; ignore = (:front, :tail)),
    ),
)

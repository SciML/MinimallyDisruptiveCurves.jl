using Documenter
using Literate
using MinimallyDisruptiveCurves

# 1. Setup paths for Literate
example_input_dir = joinpath(@__DIR__, "examples")
example_output_dir = joinpath(@__DIR__, "src", "examples")
mkpath(example_output_dir)

# 2. Convert the example script to markdown
# documenter=true tells Literate to evaluate the code and embed the output!
Literate.markdown(
    joinpath(example_input_dir, "01_basic_mass_spring.jl"),
    example_output_dir;
    documenter = true
)

Literate.markdown(
    joinpath(example_input_dir, "02_transforming_costs.jl"),
    example_output_dir;
    documenter = true
)

Literate.markdown(
    joinpath(example_input_dir, "03_basic_lotka_volterra.jl"),
    example_output_dir;
    documenter = true
)

Literate.markdown(
    joinpath(example_input_dir, "04_lotka_volterra_with_optim.jl"),
    example_output_dir;
    documenter = true
)

Literate.markdown(
    joinpath(example_input_dir, "05_nfkb.jl"),
    example_output_dir;
    documenter = true
)


# 3. Build the documentation
makedocs(
    sitename = "MinimallyDisruptiveCurves.jl Documentation",
    modules = [MinimallyDisruptiveCurves],
    warnonly = [:missing_docs],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Basic Mass-Spring" => "examples/01_basic_mass_spring.md",
            "Mass-Spring with Transforms" => "examples/02_transforming_costs.md",
            "Basic Lotka-Volterra" => "examples/03_basic_lotka_volterra.md",
            "MTK Lotka-Volterra" => "examples/04_lotka_volterra_with_optim.md",
            "NFKB model" => "examples/05_nfkb.md",
        ],
    ]

)
deploydocs(;
    repo = "github.com/SciML/MinimallyDisruptiveCurves.jl",
    devbranch = "master",
)

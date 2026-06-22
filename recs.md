
### 1. Problem and Solution Types
In SciML, a type that holds the equations, initial conditions, and  parameters to be solved is typically named `<Name>Problem`, and the  of the solve is `<Name>Solution`.

* **`MDCSystem`** $\rightarrow$ **`MDCProblem`**
  * *Reasoning*: `MDCSystem` currently encapsulates the cost function,  initial state, momentum, and names. This matches the definition of a  mathematical *problem* (like `ODEProblem` or `OptimizationProblem`),  rather than a declarative *system* (which in SciML usually refers to a  model definition like `ODESystem` in ModelingToolkit).
* **`MDCCurve`** $\rightarrow$ **`MDCSolution`**
  * *Reasoning*: The object returned by the solving process wraps the  positive and negative `ODESolution` trajectories. Naming it `MDCSolution`  aligns with `ODESolution`, `OptimizationSolution`, etc.

### 2. Solving and Algorithm API
SciML packages generally prefer lowercase functions for the solving  interface and allow the user to pass an algorithm type.

* **`MDCSolve`** $\rightarrow$ **`solve(::MDCProblem, alg; kwargs...)`**  or **`mdcsolve`**
  * *Reasoning*: `MDCSolve` uses PascalCase for a function. If you want  hook into the common SciML `solve(prob, alg)` dispatch, you can extend ` `SciMLBase.solve`. If keeping it internal, lowercase `mdcsolve` is  preferred.
* **Algorithm choice**: Currently, `Tsit5()` is hardcoded inside `MDCSolve `MDCSolve`. Consider taking `alg=Tsit5()` as a keyword argument so users  can pass other ODE solvers (e.g., `Rodas4()` for stiff cost functions).

### 3. Abstract Types
Introduce abstract types to allow for extensibility and testability,  is standard practice in SciML.

* **`MDCSystem` / `MDCCurve`** $\rightarrow$ Add **`AbstractMDCProblem`**  and **`AbstractMDCSolution`**.
* **`AbstractCost`** $\rightarrow$ **`AbstractCostFunction`** (or keep ` `AbstractCost`, but ensure it mirrors the naming of `CostFunction`).

### 4. Function Spelling and Casing
SciML strongly prefers American English spelling and standard Julia ` `snake_case` for functions. Mutation functions should end with `!`.

* **`initialise_lambda`** $\rightarrow$ **`initialize_lambda`**
  * *Reasoning*: American English is the standard in Julia base and SciML  (`initialize` instead of `initialise`).
* **`vectorfield`** $\rightarrow$ **`vectorfield!`** or **`mdc_vectorfield **`mdc_vectorfield!`**
  * *Reasoning*: The returned function mutates the `du` buffer in-place.  Adding `!` communicates the mutation convention to the user.
* **`mdc_dHdu_residual`** $\rightarrow$ **`hamiltonian_residual`** or **` **`dHdu_residual`**
  * *Reasoning*: Drop the `mdc_` prefix if the function is exported from  the package, as module qualification (`MinimallyDisruptiveCurves. (`MinimallyDisruptiveCurves.hamiltonian_residual`) handles namespacing.

### 5. Callback Factories
The `mdc_` prefix is largely redundant if these are exported from the ` `MinimallyDisruptiveCurves` module. Standard SciML packages (like ` `DiffEqCallbacks`) don't use package prefixes on exported functions.

* **`mdc_safety_callback`** $\rightarrow$ **`safety_callback`**
* **`mdc_bounds_callback`** $\rightarrow$ **`bounds_callback`**
* **`mdc_verbose_callbacks`** $\rightarrow$ **`verbose_callbacks`**
* **`mdc_momentum_readjustment`** $\rightarrow$ **` **`momentum_readjustment_callback`** (or just `momentum_readjustment`)

### Summary of API Mapping

| Current Name | Recommended Name | Rationale |
|---|---|---|
| `MDCSystem` | `MDCProblem` | Holds problem definition, aligns with ` `ODEProblem` |
| `MDCCurve` | `MDCSolution` | Holds results, aligns with `ODESolution` |
| `MDCSolve` | `mdcsolve` or `solve` | Lowercase for functions |
| `MDCWorkspace` | `MDCWorkspace` or `MDCCache` | Acceptable, but `Cache`  is common in SciML internals |
| `initialise_lambda` | `initialize_lambda` | US spelling standard |
| `vectorfield` | `vectorfield!` | Conveys in-place mutation |
| `mdc_safety_callback` | `safety_callback` | Remove module prefix for  exported API |
| `mdc_dHdu_residual` | `hamiltonian_residual` | More descriptive, no  prefix |
| `CostFunction` | `CostFunction` | Fine, or `MDCCostFunction` if  is a concern |


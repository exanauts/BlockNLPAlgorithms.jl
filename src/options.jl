abstract type AbstractOptions end

function set_options!(opt::AbstractOptions, option_dict::Dict{Symbol,Any})
    for (key, val) in option_dict
        hasproperty(opt, key) || continue
        T = fieldtype(typeof(opt), key)
        val isa T ? setproperty!(opt, key, val) :
        @error "Option $(key) needs to be of type $(T), hence discarded."
        pop!(option_dict, key)
    end
end

function set_options!(opt::AbstractOptions, option_dict::Dict{Symbol,Any}, kwargs)
    !isempty(kwargs) && (
        for (key, val) in kwargs
            option_dict[key] = val
        end
    )
    set_options!(opt, option_dict)
end

@kwdef mutable struct Options <: AbstractOptions
    # Output options
    verbosity::Int = 1
    output_file::String = "" # TODO: Provide an option to store output in a file.

    # Initialization options
    primal_start::Vector{Float64}
    dual_start::Vector{Float64}

    # Iteration options
    step_size::Float64 = 1e-1
    damping_param::Float64 = 1
    update_scheme::String = "GAUSS_SEIDEL"
    subproblem_solver::String = "MadNLP"

    # Termination options
    feas_tol::Float64 = 1e-4
    obj_conv_tol::Float64 = 1e-4
    max_iter::Int = 100
    max_wall_time::Float64 = 600.0
end
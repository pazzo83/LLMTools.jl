abstract type Runnable end

struct RunnableChain{R1<: Runnable, R2 <: Runnable} <: Runnable
    first_runnable::R1
    last_runnable::R2
end

function compose(runnable1::Runnable, runnable2::Runnable)
    return RunnableChain(runnable1, runnable2)
end

function invoke(chain::RunnableChain; kwargs...)
    input1 = invoke(chain.first_runnable; kwargs...)
    return invoke(chain.last_runnable; input1...)
end

# Wrapper for functions to adapt them as Runnables
struct FuncWrapper{T} <: Runnable
    func::T
end

function invoke(func_wrapper::FuncWrapper; kwargs...)
    args = values(kwargs)
    return func_wrapper.func(args...)
end

# Runnable for parallel ops
struct RunnableParallel{T} <: Runnable
    parallel_ops::T
end

function invoke(runnable_parallel::RunnableParallel; input...)
    ops_keys = keys(runnable_parallel.parallel_ops)
    ops_vals = (invoke(x; input...) for x in values(runnable_parallel.parallel_ops))
    
    return NamedTuple{ops_keys}(ops_vals)
end
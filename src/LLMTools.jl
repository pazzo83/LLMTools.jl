using LibPQ, Tables, JSON3
import OpenAI: AzureProvider, create_chat, create_embeddings, openai_request

import Base.(|>)
const compose = (|>)

module Pgvector
    convert(v::AbstractVector{T}) where T<:Real = string("[", join(v, ","), "]")

    parse(v::String) = map(x -> Base.parse(Float32, x), split(v[2:end-1], ","))
end

const MODEL_COST_REGISTRY = Dict{String, Tuple{Float64, Float64}}(
    "gpt-35-turbo-16k" => (5.0e-7, 1.5e-6)
)

include("runnable.jl")
include("messages.jl")
include("templates.jl")
include("llm.jl")
include("retrievers.jl")
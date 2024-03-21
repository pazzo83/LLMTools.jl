abstract type AbstractMessage end
abstract type AbstractChatMessage <: AbstractMessage end # with text-based content
abstract type AbstractDataMessage <: AbstractMessage end # with data-based content, eg, embeddings


const RESERVED_KWARGS = [
    :http_kwargs,
    :api_kwargs,
    :conversation,
    :return_all,
    :dry_run,
    :image_url,
    :image_path,
    :image_detail,
    :model
]


# utils
function _extract_handlebar_variables(s::AbstractString)
    Symbol[Symbol(m[1]) for m in eachmatch(r"\{\{([^\}]+)\}\}", s)]
end

function _extract_handlebar_variables(vect::Vector{Dict{String, <:AbstractString}})
    unique([_extract_handlebar_variables(v) for d in vect for (k, v) in d if k == "text"])
end


Base.@kwdef struct MetadataMessage{T <: AbstractString} <: AbstractChatMessage
    content::T
    description::String = ""
    version::String = "1"
    source::String = ""
    _type::Symbol = :metadatamessage
end


Base.@kwdef struct SystemMessage{T <: AbstractString} <: AbstractChatMessage
    content::T
    variables::Vector{Symbol} = _extract_handlebar_variables(content)
    _type::Symbol = :systemmessage
    SystemMessage{T}(c, v, t) where {T <: AbstractString} = new(c, v, t)
end

function SystemMessage(content::T,
        variables::Vector{Symbol},
        _type::Symbol) where {T <: AbstractString}
    not_allowed_kwargs = intersect(variables, RESERVED_KWARGS)
    @assert length(not_allowed_kwargs)==0 "Error: Some placeholders are invalid: $(join(not_allowed_kwargs, ","))"
    return SystemMessage{T}(content, variables, _type)
end


Base.@kwdef struct UserMessage{T <: AbstractString} <: AbstractChatMessage
    content::T
    variables::Vector{Symbol} = _extract_handlebar_variables(content)
    _type::Symbol = :usermessage
    UserMessage{T}(c, v, t) where {T <: AbstractString} = new(c, v, t)
end

function UserMessage(content::T,
        variables::Vector{Symbol},
        _type::Symbol) where {T <: AbstractString}
    not_allowed_kwargs = intersect(variables, RESERVED_KWARGS)
    @assert length(not_allowed_kwargs)==0 "Error: Some placeholders are invalid: $(join(not_allowed_kwargs, ","))"
    return UserMessage{T}(content, variables, _type)
end


# AI Message
Base.@kwdef struct AIMessage{T <: Union{AbstractString, Nothing}} <: AbstractChatMessage
    content::T = nothing
    status::Union{Int, Nothing} = nothing
    tokens::Tuple{Int, Int} = (-1, -1)
    elapsed::Float64 = -1.0
    cost::Union{Nothing, Float64} = nothing
    log_prob::Union{Nothing, Float64} = nothing
    finish_reason::Union{Nothing, String} = nothing
    run_id::Union{Nothing, Int} = Int(rand(Int16))
    sample_id::Union{Nothing, Int} = nothing
    _type::Symbol = :aimessage
end


# content only constructor
function (MSG::Type{<:AbstractChatMessage})(prompt::AbstractString)
    MSG(; content = prompt)
end


isusermessage(m::UserMessage) = true
isusermessage(m::AbstractMessage) = false

issystemmessage(m::SystemMessage) = true
issystemmessage(m::AbstractMessage) = false

isaimessage(m::AIMessage) = true
isaimessage(m::AbstractMessage) = false


# helpful message accessors
function last_message(conversation::AbstractVector{<:AbstractMessage})
    length(conversation) == 0 ? nothing : conversation[end]
end

function last_output(conversation::AbstractVector{<:AbstractMessage})
    msg = last_message(conversation)
    return isnothing(msg) ? nothing : msg.content
end

last_message(msg::AbstractMessage) = msg
last_output(msg::AbstractMessage) = msg.content
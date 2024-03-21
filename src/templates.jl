# main functions
function render end

## Prompt Schema
abstract type AbstractPromptSchema end

struct NoSchema <: AbstractPromptSchema end

abstract type AbstractOpenAISchema <: AbstractPromptSchema end

struct OpenAISchema <: AbstractOpenAISchema end

struct ChatPromptTemplate{S <: AbstractPromptSchema} <: Runnable
    schema::S
    messages::Vector{Union{UserMessage, SystemMessage}}
end


const ALLOWED_PROMPT_TYPE = Union{
    AbstractString,
    AbstractMessage,
    Vector{<:AbstractMessage}
}


function render(schema::NoSchema,
    messages::Vector{<:AbstractMessage};
    conversation::AbstractVector{<:AbstractMessage} = AbstractMessage[],
    replacement_kwargs...)

    # copy conversation to avoid mutation the original
    conversation = copy(conversation)
    count_system_msg = count(issystemmessage, conversation)

    # TODO: concat multiple system messages together (2nd pass)

    # replace any handelbar variables in messages
    for msg in messages
        if msg isa Union{SystemMessage, UserMessage}
            replacements = ["{{$(key)}}" => value
                for (key, value) in pairs(replacement_kwargs)
                    if key in msg.variables]
            
            # Rebuild message with replaced content
            MSGTYPE = typeof(msg)
            new_msg = MSGTYPE(;
                # unpack the type to replace only the content field
                [(field, getfield(msg, field)) for field in fieldnames(typeof(msg))]...,
                content = replace(msg.content, replacements...))
            if issystemmessage(msg)
                count_system_msg += 1
                
                # move to front
                pushfirst!(conversation, new_msg)
            else
                push!(conversation, new_msg)
            end
        elseif isaimessage(msg)
            # no replacements
            push!(conversation, msg)
        else
            # Note: ignores any DataMessage
            @warn "Unexpected message type: $(typeof(msg)).  Skipping"
        end
    end

    ## Multiple system prompts are not allowed
    (count_system_msg > 1) && throw(ArgumentError("Only one system message is allowed."))

    # add default system prompt if not provided
    (count_system_msg == 0) && pushfirst!(conversation, SystemMessage("Act as a helpful AI assistant"))

    return conversation
end


function render(schema::AbstractOpenAISchema,
    messages::Vector{<:AbstractMessage};
    conversation::AbstractVector{<:AbstractMessage} = AbstractMessage[],
    kwargs...)
    ## first pass - keep the message types but make the replacements provided in `kwargs`
    messages_replaced = render(NoSchema(), messages; conversation, kwargs...)

    ## Second pass - convert to OpenAI schema
    conversation = Dict{String, Any}[]

    # replace any handlebar variables in the messages
    for msg in messages_replaced
        role = if issystemmessage(msg)
            "system"
        elseif isusermessage(msg)
            "user"
        elseif isaimessage(msg)
            "assistant"
        end
        
        content = msg.content
        
        push!(conversation, Dict("role" => role, "content" => content))
    end

    return conversation
end


function render(prompt_template::ChatPromptTemplate;
    conversation::AbstractVector{<:AbstractMessage} = AbstractMessage[],
    kwargs...)
    render(prompt_template.schema, prompt_template.messages; conversation, kwargs...)
end

invoke(template::ChatPromptTemplate; kwargs...) = (messages = render(template; kwargs...),)
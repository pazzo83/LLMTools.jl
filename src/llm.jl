## LLM Interfaces

_get_provider(api_key::String, base_url::String, api_version::String) = AzureProvider(api_key, base_url, api_version)

## Embedding Wrapper
struct AzureOpenAIEmbeddings
    provider::AzureProvider
    model_id::String
end

function AzureOpenAIEmbeddings(api_key::String, base_url::String, api_version::String, model_id::String)
    provider = _get_provider(api_key, base_url, api_version)
    return AzureOpenAIEmbeddings(provider, model_id)
end

# Chat Model Wrapper
struct AzureOpenAIChatModel <: Runnable
    provider::AzureProvider
    model_id::String
end

function AzureOpenAIChatModel(api_key::String, base_url::String, api_version::String, model_id::String)
    provider = _get_provider(api_key, base_url, api_version)
    return AzureOpenAIChatModel(provider, model_id)
end


function invoke(model::AzureOpenAIChatModel; messages::Vector{Dict{String, Any}} = Vector{Dict{String, Any}}[])
    time = @elapsed raw_response = create_chat(model, messages)
    
    return (ai_response = response_to_message(OpenAISchema(), raw_response, time),)
end


function create_embeddings(
    embedding_model::AzureOpenAIEmbeddings,
    input,
    model_id::String,
    http_kwargs::NamedTuple = NamedTuple(),
    kwargs...)
    
    provider = embedding_model.provider
    return openai_request(
        "embeddings",
        provider;
        method = "POST",
        http_kwargs,
        model = model_id,
        input,
        query = Dict("api-version" => provider.api_version),
        kwargs...
    )
end


function create_chat(model::AzureOpenAIChatModel,
    messages::Vector{Dict{String, Any}};
    http_kwargs::NamedTuple = NamedTuple(),
    kwargs...)
    return create_chat(model.provider,
        model.model_id,
        messages;
        http_kwargs,
        query = Dict("api-version" => model.provider.api_version),
        kwargs...
    )
end


function compute_cost(model_name::String, prompt_tokens::Int, gen_tokens::Int)
    model_cost = get(MODEL_COST_REGISTRY, model_name, (0.0, 0.0))
    return (model_cost[1] * prompt_tokens) + (model_cost[2] * gen_tokens)
end

function response_to_message(schema::AbstractOpenAISchema, response, time::Float64 = 0.0, run_id::Int = Int(rand(Int32)))
    msg_data = response.response
    tokens_prompt = get(msg_data, :usage, Dict(:prompt_tokens => 0))[:prompt_tokens]
    tokens_completion = get(msg_data, :usage, Dict(:completion_tokens => 0))[:completion_tokens]
    call_cost = compute_cost(msg_data[:model], tokens_prompt, tokens_completion)
    
    choice = msg_data[:choices][begin]
    msg = AIMessage(
        content = choice[:message][:content] |> strip,
        status = Int(response.status),
        tokens = (tokens_prompt, tokens_completion),
        elapsed = time,
        cost = call_cost,
        finish_reason = get(choice, :finish_reason, nothing),
        run_id = run_id
    )
end 
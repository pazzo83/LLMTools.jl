# QUERIES
const SEARCH_QUERY = "SELECT document::varchar, cmetadata::json, embedding <=> \$1 as distance FROM langchain_pg_embedding ORDER BY distance ASC LIMIT \$2;"

struct PGVectorRetriever <: Runnable
    connection_string::String
    embedding_model::AzureOpenAIEmbeddings
    collection_name::String
end

struct Document
    page_content::String
    metadata::Dict{Symbol, Any}
end

Document(page_content::String, metadata::String) = Document(page_content, JSON3.read(metadata))

get_pg_connection_string(host::String, port::String, username::String, password::String, database::String) = begin
    return "postgresql://$username:$password@$host:$port/$database"
end


function similarity_search(pgvector::PGVectorRetriever, doc::String, k::Int)
    # get embedding
    resp = create_embeddings(pgvector.embedding_model, [doc], "text-embedding-3-small")
    embedding = resp.response.data[begin].embedding
    embedding_str = Pgvector.convert(embedding)
    
    # look up in db
    conn = LibPQ.Connection(pgvector.connection_string)
    result = execute(conn, SEARCH_QUERY, [embedding_str, k])
    data = columntable(result)
    return Document.(data.document, data.cmetadata)
end


invoke(retriever::PGVectorRetriever; query::String = "", k::Int = 4) = (docs = similarity_search(retriever, query, k), )
# DEFAULT_EMBEDDING_DIMENSION: The default dimension
# size for vector embeddings.
DEFAULT_EMBEDDING_DIMENSION = 128

# DEFAULT_CHUNK_SIZE_IN_TOKENS: The default size of text
# chunks in tokens for processing.
DEFAULT_CHUNK_SIZE_IN_TOKENS = 512

# DEFAULT_LLAMA_STACK_URL: The default URL for
# connecting to the Llama Stack service.
DEFAULT_LLAMA_STACK_URL = "http://localhost:8321"

# DEFAULT_LLAMA_STACK_WAITING_RETRIES: The default number
# of retries for waiting operations in Llama Stack.
DEFAULT_LLAMA_STACK_WAITING_RETRIES = 2

# DEFAULT_LLAMA_STACK_RETRY_DELAY: The default delay in
# seconds between retries in Llama Stack.
DEFAULT_LLAMA_STACK_RETRY_DELAY = 5

# DEFAULT_HTTP_REQUEST_TIMEOUT: The default timeout in
# seconds for HTTP requests.
DEFAULT_HTTP_REQUEST_TIMEOUT = 60

# DEFAULT_INFERENCE_MODEL: The default inference model
# used by the RAGService.
DEFAULT_INFERENCE_MODEL = "ollama/llama3.2:3b"

# DEFAULT_INGESTION_CONFIG: The default path to the ingestion
# configuration file.
DEFAULT_INGESTION_CONFIG = "config/ingestion_config.yaml"

# DEFAULT_INITIAL_CONTENT: The default initial content template
# for summarization responses.
DEFAULT_INITIAL_CONTENT = """Summarize that the user query is classified as
{department_display_name}, along with any answers provided by the LLM
for the question, and include that we are responding to submission_id
{state_sub_id}. Finally, mention a GitHub issue will be opened
for follow up.
"""

# RAG_PROMPT_TEMPLATE: The prompt template used for
# retrieval-augmented generation (RAG) responses.
RAG_PROMPT_TEMPLATE = """Based on the relevant documents in the knowledge base,
please help with the following {department_display_name} query:

{user_input}

Please provide a helpful response based on the documents found. If no relevant
documents are found, provide general guidance."""

# DEFAULT_INGESTION_CONFIG_PATHS: The default paths to look for
# the ingestion configuration file.
DEFAULT_INGESTION_CONFIG_PATHS = [
    "ingestion-config.yaml",
    "/config/ingestion-config.yaml",
]

# DEFAULT_RAG_METADATA_FILE_PATHS: The default file location
# for storing RAG metadata.
DEFAULT_RAG_METADATA_FILE_PATHS = [
    "rag_file_metadata.json",
    "/config/rag_file_metadata.json",
]

# PIPELINE_CATEGORIES: The list of pipeline categories
PIPELINE_CATEGORIES = ["legal", "techsupport", "hr", "sales", "procurement"]

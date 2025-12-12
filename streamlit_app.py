import asyncio
import os
import time
import uuid
from typing import Any

import httpx
import streamlit as st

from src.constants import (
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_INGESTION_CONFIG,
    DEFAULT_LLAMA_STACK_URL,
)
from src.ingest import IngestionService
from src.responses import RAGService
from src.types import Pipeline
from src.utils import logger
from src.workflow import Workflow, submission_states

API_KEY = os.getenv("OPENAI_API_KEY", "not applicable")
INFERENCE_SERVER_OPENAI = os.getenv(
    "LLAMA_STACK_SERVER_OPENAI", "http://localhost:8321/v1/openai/v1"
)
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", DEFAULT_INFERENCE_MODEL)
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "ollama/llama-guard3:8b")
MCP_TOOL_MODEL = os.getenv("MCP_TOOL_MODEL", "ollama/llama3.2:3b")
GIT_TOKEN = os.getenv("GIT_TOKEN", "not applicable")
GITHUB_URL = os.getenv("GITHUB_URL", "not applicable")
GITHUB_ID = os.getenv("GITHUB_ID", "not applicable")
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", DEFAULT_LLAMA_STACK_URL)
INGESTION_CONFIG = os.getenv("INGESTION_CONFIG", DEFAULT_INGESTION_CONFIG)
RAG_FILE_METADATA = os.getenv("RAG_FILE_METADATA", "rag_file_metadata.json")


@st.cache_resource
def get_config() -> "dict[str, str]":
    """
    gets configuration (cached)"""
    return {
        "inference_server": INFERENCE_SERVER_OPENAI,
        "llama_stack_url": LLAMA_STACK_URL,
        "ingestion_config": INGESTION_CONFIG,
        "rag_metadata": RAG_FILE_METADATA,
    }


@st.cache_resource
def initialize_workflow(_pipelines: "list[Pipeline]") -> "tuple[Any, RAGService]":
    """
    initializes workflow with pipelines from ingestion (cached)
    Returns tuple of (compiled_workflow, rag_service)
    Note: _pipelines is prefixed with _ to avoid hashing by Streamlit
    """
    rag_service = RAGService(
        llama_stack_url=LLAMA_STACK_URL,
        ingestion_config_path=INGESTION_CONFIG,
        file_metadata_path=RAG_FILE_METADATA,
        pipelines=_pipelines,
    )
    if not rag_service.initialize():
        logger.warning("RAG Service initialization failed.")

    # create Workflow instance and compile it
    workflow_builder = Workflow(rag_service=rag_service)
    compiled_workflow = workflow_builder.make_workflow(
        tools_llm=INFERENCE_MODEL,
        git_token=GIT_TOKEN,
        github_url=GITHUB_URL,
        guardrail_model=GUARDRAIL_MODEL,
    )

    logger.info("✓ Workflow and RAG Service initialized and cached")
    return compiled_workflow, rag_service


def get_or_create_event_loop() -> "Any":
    """
    gets an existing event loop from session state or create a new one
    """
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.event_loop)
    return st.session_state.event_loop


def get_tasks_dict() -> "Any":
    """
    gets tasks dictionary from session state
    """
    if "async_tasks" not in st.session_state:
        st.session_state.async_tasks = {}
    return st.session_state.async_tasks


def get_ingestion_state() -> "dict[str, Any]":
    """
    gets or initializes ingestion state
    """
    if "ingestion_state" not in st.session_state:
        st.session_state.ingestion_state = {
            "status": "pending",  # pending, running, completed, error, skipped
            "message": "",
            "ingested_count": 0,
            "pipelines": None,
        }
    return st.session_state.ingestion_state


def count_vector_stores() -> "int":
    """
    count the number of vector stores in the database
    """
    try:
        from llama_stack_client import LlamaStackClient

        client = LlamaStackClient(base_url=LLAMA_STACK_URL)
        vector_stores = client.vector_stores.list() or []
        vector_store_list = list(vector_stores)
        count = len(vector_store_list)
        logger.debug(f"Found {count} vector stores in database")
        return count
    except Exception as e:
        logger.warning(f"Failed to count vector stores: {e}")
        return 0


def skip_ingestion() -> "None":
    """
    skips the ingestion pipeline but load pipelines from config
    """
    ingestion_state = get_ingestion_state()

    try:
        logger.info("Skipping ingestion, loading pipelines from config")
        ingestion_service = IngestionService(INGESTION_CONFIG)
        pipelines = ingestion_service.pipelines

        # Count existing vector stores
        vector_store_count = count_vector_stores()

        ingestion_state["status"] = "skipped"
        ingestion_state["message"] = (
            f"Ingestion skipped - loaded {len(pipelines)} pipelines from config"
        )
        ingestion_state["pipelines"] = pipelines
        ingestion_state["vector_store_count"] = vector_store_count
        logger.info(
            f"Ingestion skipped: loaded {len(pipelines)} pipelines from config, "
            f"{vector_store_count} vector stores in database"
        )
    except Exception as e:
        logger.error(f"Failed to load pipelines from config: {e}")
        ingestion_state["status"] = "error"
        ingestion_state["message"] = f"Failed to load pipelines: {str(e)}"
        ingestion_state["pipelines"] = []


async def run_ingestion_pipeline() -> "None":
    """
    runs the ingestion pipeline asynchronously
    """
    ingestion_state = get_ingestion_state()

    try:
        ingestion_state["status"] = "running"
        ingestion_state["message"] = "Checking llama-stack server availability..."
        logger.info("Starting Ingestion Service...")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{LLAMA_STACK_URL}/v1/health")
                if response.status_code != 200:
                    raise Exception(
                        f"Llama-stack server returned status {response.status_code}"
                    )
                logger.info("✓ Llama-stack server is available")
        except Exception as e:
            raise Exception(
                f"Llama-stack server not available at {LLAMA_STACK_URL}. "
                f"Please start the server first: {e}"
            )

        ingestion_state["message"] = "Initializing ingestion service..."

        # run the ingestion pipeline in a thread to avoid blocking the event loop
        ingestion_service = await asyncio.to_thread(IngestionService, INGESTION_CONFIG)

        ingestion_state["message"] = (
            "Running ingestion pipeline (this may take a while)..."
        )
        logger.info("Running ingestion pipeline...")
        ingested_items = await asyncio.to_thread(ingestion_service.run)

        # store pipelines from ingestion service
        ingestion_state["pipelines"] = ingestion_service.pipelines

        vector_store_count = await asyncio.to_thread(count_vector_stores)

        ingestion_state["status"] = "completed"
        ingestion_state["ingested_count"] = len(ingested_items) if ingested_items else 0
        ingestion_state["vector_store_count"] = vector_store_count
        ingestion_state["message"] = (
            "Ingestion completed successfully! "
            f"Processed {ingestion_state['ingested_count']} items."
        )
        logger.info(
            f"Ingestion completed: {ingestion_state['ingested_count']} "
            f"items processed, {vector_store_count} vector stores in database"
        )

    except Exception as e:
        ingestion_state["status"] = "error"
        error_msg = str(e)
        ingestion_state["message"] = f"Ingestion failed: {error_msg[:200]}"
        logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)


def start_ingestion_if_needed() -> "None":
    """
    Start ingestion pipeline if not already started"""
    ingestion_state = get_ingestion_state()

    if ingestion_state["status"] == "pending":
        loop = get_or_create_event_loop()
        tasks = get_tasks_dict()

        ingestion_task = loop.create_task(run_ingestion_pipeline())
        tasks["__ingestion__"] = ingestion_task
        logger.info("Ingestion pipeline task submitted")


async def run_workflow_task(
    workflow: "Workflow", question: "str", submission_id: "str"
) -> "None":
    """Async task to run a single workflow"""
    try:
        logger.info(f"Starting workflow task for submission {submission_id}")

        # Execute workflow using asyncio.to_thread to avoid blocking
        result = await asyncio.to_thread(
            workflow.invoke,
            {
                "input": question,
                "submission_id": submission_id,
                "messages": [],
                "decision": "",
                "namespace": "",
                "data": "",
                "mcp_output": "",
                "github_issue": "",
                "rag_sources": [],
                "workflow_complete": False,
                "classification_message": "",
                "agent_timings": {},
                "rag_query_time": 0.0,
                "active_agent": "",
            },
        )

        # Update submission_states with the final workflow result
        submission_states[submission_id] = result
        logger.info(
            f"Workflow task completed for submission {submission_id}: "
            f"decision={result.get('decision')}, "
            f"complete={result.get('workflow_complete')}"
        )
    except Exception as e:
        logger.error(f"Workflow task failed for submission {submission_id}: {e}")
        submission_states[submission_id] = {
            "input": question,
            "submission_id": submission_id,
            "decision": "error",
            "classification_message": f"Error: {str(e)[:200]}",
            "workflow_complete": True,
            "mcp_output": "",
            "github_issue": "",
            "rag_sources": [],
            "messages": [],
            "namespace": "",
            "data": "",
            "agent_timings": {},
            "rag_query_time": 0.0,
            "active_agent": "",
        }


def progress_event_loop() -> "None":
    """
    progress the event loop to advance all pending tasks without blocking UI
    """
    loop = get_or_create_event_loop()
    tasks = get_tasks_dict()

    pending_tasks = [task for task in tasks.values() if not task.done()]

    if pending_tasks:
        try:
            # Just tick the event loop once without blocking
            # This allows tasks to make progress without freezing the UI
            loop.run_until_complete(asyncio.sleep(0))
        except Exception as e:
            logger.error(f"Error progressing event loop: {e}")


def submit_workflow_task(
    workflow: "Workflow", question: "str", submission_id: "str"
) -> "None":
    """Submit a new workflow task to the event loop"""
    loop = get_or_create_event_loop()
    tasks = get_tasks_dict()

    task = loop.create_task(run_workflow_task(workflow, question, submission_id))
    tasks[submission_id] = task

    logger.info(f"Submitted workflow task for {submission_id}")


def main():
    st.set_page_config(
        page_title="Agentic AI Workflow",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    progress_event_loop()

    ingestion_state = get_ingestion_state()

    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown(f"**Inference Model:** `{INFERENCE_MODEL}`")
        st.markdown(f"**Guardrail Model:** `{GUARDRAIL_MODEL}`")
        st.markdown(f"**MCP Tool Model:** `{MCP_TOOL_MODEL}`")
        st.divider()

        st.subheader("📦 Ingestion Status")

        if "vector_store_count" not in ingestion_state:
            ingestion_state["vector_store_count"] = count_vector_stores()

        vector_store_count = ingestion_state.get("vector_store_count", 0)

        if ingestion_state["status"] == "pending":
            st.warning("⏸️ Waiting for user action")
        elif ingestion_state["status"] == "running":
            st.info("⏳ Running...")
        elif ingestion_state["status"] == "completed":
            st.success(f"✅ Completed ({ingestion_state['ingested_count']} items)")
        elif ingestion_state["status"] == "skipped":
            st.info("⏭️ Skipped")
        elif ingestion_state["status"] == "error":
            st.error("❌ Failed")

        if vector_store_count > 0:
            st.metric("Vector Stores in Database", vector_store_count)
        st.divider()

    if ingestion_state["status"] in ("pending", "running"):
        st.title("🤖 Agentic AI Workflow - Initializing")

        if ingestion_state["status"] == "running":
            st.info("⏳ Running data ingestion pipeline... Please wait.")
        else:
            st.info("📋 Data ingestion pipeline is ready to run.")
            st.markdown(
                """
            **Choose an option:**
            - **Run Ingestion**: Process documents and create vector stores
            (recommended for first time)
            - **Skip Ingestion**: Load pipeline configuration only
            (use if data is already ingested)
            """
            )

            col1, col2, _ = st.columns([1, 1, 3])
            with col1:
                if st.button("▶️ Run Ingestion", type="primary"):
                    start_ingestion_if_needed()
                    st.rerun()
            with col2:
                if st.button("⏭️ Skip Ingestion", type="secondary"):
                    skip_ingestion()
                    st.rerun()
            return

        time.sleep(0.5)
        st.rerun()
        return

    pipelines = ingestion_state.get("pipelines")
    if pipelines is None:
        # parse pipelines from config file without running ingestion
        logger.info("Pipelines not available, parsing from ingestion config")
        ingestion_service = IngestionService(INGESTION_CONFIG)
        pipelines = ingestion_service.pipelines
        logger.info(f"Loaded {len(pipelines)} pipelines from config")

    # Initialize workflow only if not already in session state
    if "workflow" not in st.session_state:
        workflow, rag_service = initialize_workflow(pipelines)
        st.session_state.workflow = workflow
        vector_store_count = len(rag_service.all_vector_store_ids)
        ingestion_state["vector_store_count"] = vector_store_count
        logger.info(f"Vector stores in database: {vector_store_count}")
    else:
        # Workflow already initialized, just use it from session state
        workflow = st.session_state.workflow

    tasks = get_tasks_dict()
    has_active_tasks = any(
        not task.done() for task_id, task in tasks.items() if task_id != "__ingestion__"
    )

    with st.sidebar:
        st.subheader("📊 Active Submissions")
        if "active_submissions" not in st.session_state:
            st.session_state.active_submissions = []

        # display active submissions
        if st.session_state.active_submissions:
            for sub_id in st.session_state.active_submissions:
                is_complete = submission_states.get(sub_id, {}).get("workflow_complete")
                decision = submission_states.get(sub_id, {}).get("decision", "").lower()

                # Determine status icon: error states (error/unsafe/unknown) show ❌
                if decision in ("error", "unsafe", "unknown"):
                    status_icon = "❌"
                elif is_complete:
                    status_icon = "✅"
                else:
                    status_icon = "⏳"

                question = submission_states.get(sub_id, {}).get("input", "")
                question_preview = (
                    question[:30] + "..." if len(question) > 30 else question
                )

                with st.expander(f"{status_icon} {sub_id[:8]}... - {question_preview}"):
                    st.markdown("**Full Submission ID:**")
                    st.code(sub_id, language=None)
        else:
            st.info("No active submissions")

        if st.button("Clear All Submissions"):
            st.session_state.active_submissions = []
            st.rerun()

    st.title("🤖 Agentic AI Workflow")
    st.markdown(
        """
    Submit your questions and track their processing in real-time.
    Multiple submissions can run concurrently.
    """
    )

    if "workflow" in st.session_state and st.session_state.workflow is not None:
        with st.expander("🔀 View Workflow Graph"):
            try:
                graph_ascii = st.session_state.workflow.get_graph().draw_ascii()
                st.code(graph_ascii, language="text")
            except Exception as e:
                st.error(f"Could not display graph: {e}")

    # Show background processing indicator but don't block UI
    if has_active_tasks:
        st.info(
            "⏳ Processing workflows in background... You can submit new questions."
        )

    tab1, tab2 = st.tabs(["📝 Submit Question", "📋 View Results"])

    with tab1:
        st.subheader("Submit a New Question")

        with st.form("question_form", clear_on_submit=True):
            question = st.text_area(
                "Enter your question:",
                placeholder=(
                    "e.g., What are the legal implications of using GPL licenses?"
                ),
                height=100,
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button(
                    "🚀 Submit", use_container_width=True
                )

            if submit_button:
                if not question.strip():
                    st.error("Please enter a question")
                else:
                    submission_id = str(uuid.uuid4())

                    if "active_submissions" not in st.session_state:
                        st.session_state.active_submissions = []
                    st.session_state.active_submissions.insert(0, submission_id)

                    submission_states[submission_id] = {
                        "input": question,
                        "submission_id": submission_id,
                        "decision": "",
                        "classification_message": "Processing...",
                        "workflow_complete": False,
                        "mcp_output": "",
                        "github_issue": "",
                        "rag_sources": [],
                        "messages": [],
                        "namespace": "",
                        "data": "",
                        "agent_timings": {},
                        "rag_query_time": 0.0,
                        "active_agent": "",
                    }

                    submit_workflow_task(workflow, question, submission_id)

                    st.success(f"✅ Submitted workflow {submission_id[:8]}...")
                    st.rerun()

    with tab2:
        st.subheader("View Submission Results")

        view_mode = st.radio(
            "Select viewing mode:",
            ["Recent Submissions", "Search by ID"],
            horizontal=True,
        )

        if view_mode == "Recent Submissions":
            if not st.session_state.get("active_submissions"):
                st.info(
                    "No submissions yet. Submit a question in "
                    "the 'Submit Question' tab."
                )
            else:
                selected_sub = st.selectbox(
                    "Select a submission:",
                    st.session_state.active_submissions,
                    format_func=lambda x: (
                        f"{x[:8]}... - {
                            submission_states.get(x, {}).get('input', 'Unknown')[:50]
                        }",
                    ),
                )

                if selected_sub:
                    display_submission_details(selected_sub)

        else:  # Search by ID
            search_id = st.text_input("Enter Submission ID:")
            if st.button("Search"):
                if search_id in submission_states:
                    display_submission_details(search_id)
                else:
                    st.error(f"Submission ID '{search_id}' not found")

    # auto-refresh when there are active tasks
    if has_active_tasks:
        time.sleep(0.5)
        st.rerun()


def display_submission_details(submission_id: "str") -> "None":
    """Display detailed information about a submission"""
    state = submission_states.get(submission_id)

    if not state:
        st.error("Submission not found")
        return

    # status indicator
    is_complete = state.get("workflow_complete", False)
    decision = state.get("decision", "")
    decision_lower = decision.lower()

    if decision_lower == "error":
        st.error("❌ Workflow Failed - Error occurred during processing")
    elif decision_lower == "unsafe":
        st.error("⚠️ Workflow Blocked - Content flagged by moderation")
    elif decision_lower == "unknown":
        st.error("❓ Workflow Failed - Unable to classify request")
    elif is_complete:
        st.success(f"✅ Workflow Complete - Decision: {decision.upper()}")
    else:
        st.info(f"⏳ Processing... Current stage: {decision or 'Classifying'}")
        if st.button("🔄 Refresh", key=f"refresh_{submission_id}"):
            st.rerun()

    st.markdown("### 📋 Submission Details")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Submission ID:** `{submission_id}`")
    with col2:
        st.markdown(f"**Status:** {decision or 'Pending'}")

    st.markdown("---")

    with st.expander("📝 Input Question", expanded=True):
        st.write(state.get("input", "N/A"))

    # Display timing information
    agent_timings = state.get("agent_timings", {})
    rag_query_time = state.get("rag_query_time", 0.0)

    if agent_timings or rag_query_time > 0:
        with st.expander("⏱️ Response Times", expanded=True):
            if agent_timings:
                st.markdown("**Agent Processing Times:**")
                for agent_name, duration in agent_timings.items():
                    st.metric(
                        label=f"{agent_name} Agent",
                        value=f"{duration:.2f}s",
                    )

            if rag_query_time > 0:
                st.markdown("**Vector Store Query Time:**")
                st.metric(
                    label="RAG Query",
                    value=f"{rag_query_time:.2f}s",
                )

            # Calculate and display total time
            total_agent_time = sum(agent_timings.values()) if agent_timings else 0
            if total_agent_time > 0:
                st.markdown("**Total Processing Time:**")
                st.metric(
                    label="Total",
                    value=f"{total_agent_time:.2f}s",
                )

    if state.get("classification_message"):
        with st.expander("🔍 Response", expanded=True):
            active_agent = state.get("active_agent", "")
            if active_agent:
                st.markdown(f"**Handled by:** {active_agent} Department")
            st.write(state["classification_message"])

    rag_sources = state.get("rag_sources", [])
    if rag_sources:
        with st.expander(
            f"📚 RAG Sources ({len(rag_sources)} documents)", expanded=False
        ):
            for i, source in enumerate(rag_sources, 1):
                st.markdown(f"**{i}.** {source.get('file_name', 'Unknown')}")
                if source.get("chunk_id"):
                    st.caption(f"Chunk: {source['chunk_id']}")

    if state.get("mcp_output"):
        with st.expander("🔧 Preliminary Diagnostics", expanded=False):
            st.code(state["mcp_output"], language="text")

    if state.get("github_issue"):
        with st.expander("🔗 GitHub Tracking Issue", expanded=True):
            st.markdown(f"[{state['github_issue']}]({state['github_issue']})")

    if st.checkbox("Show Raw State (Debug)", key=f"debug_{submission_id}"):
        st.json(state)


if __name__ == "__main__":
    main()

import os
from typing import Any

from langgraph.graph import START, StateGraph
from llama_stack_client import LlamaStackClient
from openai import OpenAI

from src.constants import (
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_INITIAL_CONTENT,
    DEFAULT_LLAMA_STACK_URL,
    RAG_PROMPT_TEMPLATE,
)
from src.methods import (
    classification_agent,
    git_agent,
    perf_agent,
    pod_agent,
    support_classification_agent,
)
from src.responses import RAGService
from src.types import WorkflowState
from src.utils import (
    extract_rag_response_text,
    logger,
    route_to_next_node,
    support_route_to_next_node,
)

submission_states: "dict[str, WorkflowState]" = {}

INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", DEFAULT_INFERENCE_MODEL)
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", DEFAULT_LLAMA_STACK_URL)


class Workflow:
    def __init__(self, rag_prompt=RAG_PROMPT_TEMPLATE):
        self.rag_prompt = rag_prompt
        self.openai_client: "OpenAI | None" = None

    def _convert_messages_to_openai_format(
        self, messages_state: "dict[str, Any]"
    ) -> "list[dict[str, str]]":
        """Convert state messages to OpenAI format"""
        messages = []
        for msg in messages_state.get("messages", []):
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, str):
                messages.append({"role": "user", "content": msg})
            elif hasattr(msg, "content"):
                # fallback to langchain message objects
                role = (
                    "assistant"
                    if hasattr(msg, "__class__") and "AI" in msg.__class__.__name__
                    else "user"
                )
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append({"role": "user", "content": str(msg)})
        return messages

    def _call_openai_llm(self, state: "WorkflowState") -> "str":
        """Call OpenAI chat completions API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        messages = self._convert_messages_to_openai_format(state)
        completion = self.openai_client.chat.completions.create(
            model=INFERENCE_MODEL, messages=messages
        )
        return completion.choices[0].message.content

    def llm_node(
        self,
        department_display_name: "str",
        state: "WorkflowState",
        additional_prompt: "str | None" = None,
        rag_service: "RAGService | None" = None,
        rag_category: "str | None" = None,
        is_terminal: "bool" = False,
    ) -> "WorkflowState":
        logger.debug(f"{department_display_name} LLM node processing")

        # check if RAG is available for this department
        use_rag = (
            rag_service is not None
            and rag_category is not None
            and hasattr(rag_service, "get_file_search_tool")
        )

        file_search_tool = None
        if use_rag and rag_service is not None:
            file_search_tool = rag_service.get_file_search_tool(rag_category)
            if file_search_tool:
                logger.info(
                    f"{department_display_name}: RAG enabled with category '{rag_category}'"
                )
            else:
                logger.info(
                    f"{department_display_name}: No vector stores for '{rag_category}', using standard LLM"
                )
                use_rag = False

        if use_rag and file_search_tool:
            # Use RAG with file_search tool via OpenAI responses API
            # uses the same openai_client.responses.create() pattern
            # as MCP tool calls
            try:
                rag_prompt = self.rag_prompt.format(
                    department_display_name=department_display_name,
                    user_input=state.input,
                )
                if additional_prompt is not None:
                    rag_prompt += f"""\n{additional_prompt}"""
                logger.info(
                    f"{department_display_name}: Making RAG-enabled response call"
                )
                rag_response = rag_service.client.responses.create(
                    model=INFERENCE_MODEL,
                    input=rag_prompt.format(
                        department_display_name=department_display_name,
                        user_input=state.input,
                    ),
                    tools=[file_search_tool],
                )

                response_text = extract_rag_response_text(rag_response)

                sources = rag_service.extract_sources_from_response(
                    rag_response, rag_category
                )
                state.rag_sources = sources
                if sources:
                    logger.info(
                        f"{department_display_name}: Found {len(sources)} source documents"
                    )

                if response_text:
                    cm = response_text
                    logger.info(f"{department_display_name}: RAG response successful")
                else:
                    logger.warning(
                        f"{department_display_name}: RAG response empty, falling back to standard LLM"
                    )
                    cm = self._call_openai_llm(state)

            except Exception as e:
                logger.error(
                    f"{department_display_name}: RAG call failed: {e}, falling back to standard LLM"
                )
                cm = self._call_openai_llm(state)
        else:
            # Standard LLM call without RAG
            cm = self._call_openai_llm(state)

        # Store the response as a message in OpenAI format
        state.messages = [{"role": "assistant", "content": cm}]
        state.classification_message = cm
        if is_terminal:
            state.workflow_complete = True
        # TODO: fix the thing below
        # sub_id = state.submission_id
        # submission_states[sub_id] = state
        return state

    def init_message(
        department_name: "str",
        department_display_name: "str",
        state: "WorkflowState",
        content_override: "str | None" = None,
    ) -> "dict[str, list[dict[str, str]]]":
        logger.info(f"init {department_name} message '{state}'")
        if content_override:
            content = content_override
        else:
            content = DEFAULT_INITIAL_CONTENT.format(
                department_display_name=department_display_name.lower(),
                state_sub_id=state.submission_id,
            )
        return {"messages": [{"role": "user", "content": content}]}

    def create_agent(
        self,
        department_name: "str",
        department_display_name: "str",
        content_override: "str | None" = None,
        custom_llm: "str" = None,
        submission_states: "dict[str, 'WorkflowState'] | None" = None,
        rag_service: "RAGService" = None,
        rag_category: "str | None" = None,
    ) -> "WorkflowState":
        """
        factory function to create department-specific agents with consistent structure.

        Args:
            department_name: Internal name for the department (e.g., 'legal', 'support')
            department_display_name: Display name (e.g., 'Legal', 'Software Support')
            content_override: Optional content to override default prompt
            custom_llm: LangChain LLM instance for standard inference
            submission_states: Dictionary to store submission states
            rag_service: Optional RAGService instance for RAG-enabled responses
            rag_category: Category name to select appropriate vector stores (e.g., 'legal', 'support')
        """

        # Use custom_llm if provided, otherwise default to topic_llm
        if custom_llm is None:
            raise ValueError("custom_llm is required")

        if submission_states is None:
            raise ValueError("submission_states is required")

        init_message = self.init_message(
            department_name,
            department_display_name,
            state,
            content_override,
        )
        state = self.llm_node(
            department_display_name,
            state,
            rag_service,
            rag_category,
        )
        agent_builder = StateGraph(WorkflowState)
        agent_builder.add_node(f"{department_name}_set_message", init_message)
        agent_builder.add_node("llm_node", state)
        agent_builder.add_edge(START, f"{department_name}_set_message")
        agent_builder.add_edge(f"{department_name}_set_message", "llm_node")
        agent_workflow = agent_builder.compile()
        logger.info(agent_workflow.get_graph().draw_ascii())

        return agent_workflow

    def make_workflow(
        self,
        topic_llm: "str",
        rag_service: "RAGService | None" = None,
    ):
        """Create and configure the overall workflow with all agents and routing.

        Args:
            topic_llm: LangChain LLM instance for classification and general inference
            openai_client: OpenAI SDK client for responses API (MCP tools, RAG file_search)
            guardrail_model: Model ID for content moderation
            mcp_tool_model: Model ID for MCP tool calls
            git_token: GitHub personal access token
            github_url: GitHub repository URL for issue creation
            github_id: GitHub user ID for issue assignment
            rag_service: Optional RAGService instance for RAG-enabled responses
            inference_model: Model ID for inference (used in RAG calls)
        """
        # TODO: What's the purpose of this?
        lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)

        # Create all department agents using the factory function
        # RAG is enabled for legal and support agents using their
        # respective vector stores
        legal_agent = self.create_agent(
            "legal",
            "Legal",
            custom_llm=topic_llm,
            submission_states=submission_states,
            rag_service=rag_service,
            rag_category="legal",
            is_terminal=True,
        )
        support_agent = self.create_agent(
            "support",
            "Software Support",
            custom_llm=topic_llm,
            submission_states=submission_states,
            rag_service=rag_service,
            rag_category="support",
            is_terminal=False,
        )

        hr_agent = self.create_agent(
            department_name="hr",
            department_display_name="Human Resources",
            custom_llm=topic_llm,
            submission_states=submission_states,
            rag_service=rag_service,
            rag_category="hr",
            additional_prompt="""
            FantaCo's benefits description is organized into such categories as:
            - bare necessities: workspace, as well as benefits such as health care, vacation or PTO, retirement plans
            - beyond the basics: music, parties and activities, food and driving services, bonuses
            - and then some minor caveats: random set of participation requirements
            if possible, try to narrow the scope of the response to the details that fall under on of those sub-sections of 
            the benefits document.
            """,
            is_terminal=True,
        )

        sales_agent = self.create_agent(
            department_name="sales",
            department_display_name="Sales",
            custom_llm=topic_llm,
            submission_states=submission_states,
            rag_service=rag_service,
            rag_category="sales",
            additional_prompt="""
            FantaCo's sales operation manual outlines policies over ten broad categories :
            - geographic territories
            - lead assignments
            - discounting
            - deal approval
            - quotas
            - compensations
            - CRMs
            - brands and communications
            - expenses
            - escalations
            - performance
            - compliance
            if possible, try to narrow the scope of the response to the details that fall under on of those sub-sections of 
            the sales document.
            """,
            is_terminal=True,
        )
        procurement_agent = self.create_agent(
            department_name="procurement",
            department_display_name="Procurement",
            custom_llm=topic_llm,
            submission_states=submission_states,
            rag_service=rag_service,
            rag_category="procurement",
            additional_prompt="""
            FantaCo's procurement policies cover :
            - competitive bidding
            - vendor evaluation and categorization
            - ethics and transparency
            - review
            - spending limits
            if possible, try to narrow the scope of the response to the details that fall under on of those sub-sections of 
            the procurement document.
            """,
            is_terminal=True,
        )

        overall_workflow = StateGraph(Workflow)
        overall_workflow.add_node("classification_agent", classification_agent)
        overall_workflow.add_node("legal_agent", legal_agent)
        overall_workflow.add_node("hr_agent", hr_agent)
        overall_workflow.add_node("sales_agent", sales_agent)
        overall_workflow.add_node("procurement_agent", procurement_agent)
        overall_workflow.add_node("support_agent", support_agent)
        overall_workflow.add_node("pod_agent", pod_agent)
        overall_workflow.add_node("perf_agent", perf_agent)
        overall_workflow.add_node("git_agent", git_agent)
        overall_workflow.add_node(
            "support_classification_agent", support_classification_agent
        )
        overall_workflow.add_edge(START, "classification_agent")
        overall_workflow.add_conditional_edges(
            "classification_agent", route_to_next_node
        )
        overall_workflow.add_edge("support_agent", "support_classification_agent")
        overall_workflow.add_conditional_edges(
            "support_classification_agent", support_route_to_next_node
        )
        overall_workflow.add_edge("pod_agent", "git_agent")
        overall_workflow.add_edge("perf_agent", "git_agent")
        workflow = overall_workflow.compile()

        logger.info(workflow.get_graph().draw_ascii())

        return workflow

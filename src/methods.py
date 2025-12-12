import time

from openai import OpenAI

from src.exceptions import AgentRunMethodParameterError
from src.models import ClassificationModel, SupportClassificationModel
from src.types import WorkflowAgentPrompts, WorkflowState
from src.utils import extract_mcp_output, logger


def classification_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    topic_llm: "str | None",
    guardrail_model: "str | None",
) -> "WorkflowState":

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    state["active_agent"] = "Classification"

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in classification agent"
        )

    if topic_llm is None:
        raise AgentRunMethodParameterError(
            "topic_llm is required in classification agent"
        )

    if guardrail_model is None:
        raise AgentRunMethodParameterError(
            "guardrail_model is required in classification agent"
        )

    # checking if the input is safe
    safety_response = openai_client.moderations.create(
        model=guardrail_model, input=str(state["input"])
    )

    for moderation in safety_response.results:
        if not moderation.flagged:
            continue

        # TODO: decide log level
        logger.info(
            f"Classification result: '{state['input']}' is flagged as '{moderation}'"
        )
        state["decision"] = "unsafe"
        state["data"] = state["input"]
        flagged_categories = [
            key
            for key, value in moderation.categories.model_extra.items()
            if value is True
        ]
        categories_str = ", ".join(flagged_categories)
        state["classification_message"] = (
            f"Classification result: '{state['input']}' "
            f"is flagged for: {categories_str}"
        )
        return state

    # Use OpenAI client for structured output with Pydantic models
    try:
        completion = openai_client.beta.chat.completions.parse(
            model=topic_llm,
            messages=[
                {
                    "role": "user",
                    "content": WorkflowAgentPrompts.CLASIFICATION_PROMPT.format(
                        state_input=state["input"]
                    ),
                }
            ],
            response_format=ClassificationModel,
        )

        classification_result = completion.choices[0].message.parsed

        # Validate classification
        if not classification_result or not hasattr(
            classification_result, "classification"
        ):
            logger.error("Failed to get structured response from the model.")
            state["decision"] = "unknown"
            state["data"] = state["input"]
            state["classification_message"] = "Unable to determine request type."
            return state

        if classification_result.classification not in (
            "legal",
            "techsupport",
            "support",
            "hr",
            "sales",
            "procurement",
            "unsafe",
            "unknown",
        ):
            logger.error(
                f"Invalid classification: {classification_result.classification}"
            )
            state["decision"] = "unknown"
            state["data"] = state["input"]
            state["classification_message"] = "Unable to determine request type."
            return state

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        state["decision"] = "unknown"
        state["data"] = state["input"]
        state["classification_message"] = f"Classification error: {str(e)[:100]}"
        return state
    logger.info(
        f"Classification result: {classification_result} for input '{state['input']}'"
    )

    state["decision"] = classification_result.classification
    state["data"] = state["input"]

    agent_end_time = time.time()
    state["agent_timings"]["Classification"] = agent_end_time - agent_start_time

    return state


def support_classification_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    topic_llm: "str | None",
) -> "WorkflowState":

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    state["active_agent"] = "Support Classification"

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support classification agent"
        )

    if topic_llm is None:
        raise AgentRunMethodParameterError(
            "model is required in support classification agent"
        )

    try:
        completion = openai_client.beta.chat.completions.parse(
            model=topic_llm,
            messages=[
                {
                    "role": "user",
                    "content": WorkflowAgentPrompts.SUPPORT_CLASIFICATION_PROMPT.format(
                        state_input=state["input"]
                    ),
                }
            ],
            response_format=SupportClassificationModel,
        )

        classification_result = completion.choices[0].message.parsed

        if not classification_result:
            logger.error(
                "Failed to get structured response from support classification."
            )
            state["decision"] = "unknown"
            state["namespace"] = ""
            return state

        logger.info(
            f"Support Classification result: {classification_result} "
            f"for input '{state['input']}'"
        )

    except Exception as e:
        logger.error(f"Support classification failed: {e}")
        state["decision"] = "unknown"
        state["namespace"] = ""
        return state
    state["namespace"] = classification_result.namespace
    state["decision"] = (
        classification_result.classification
        if classification_result.performance in ("true", "performance issue")
        else "perf"
    )
    state["data"] = state["input"]

    agent_end_time = time.time()
    state["agent_timings"]["Support Classification"] = agent_end_time - agent_start_time

    return state


def git_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    git_token: "str | None",
    tools_llm: "str | None",
    github_url: "str | None",
) -> "WorkflowState":
    logger.debug(f"git Agent request for submission: {state['submission_id']}")

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    state["active_agent"] = "Git"

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    if not git_token:
        raise AgentRunMethodParameterError("git_token is required in git agent")

    if not tools_llm:
        raise AgentRunMethodParameterError("tools_llm is required in git agent")

    openai_mcp_tool = {
        "type": "mcp",
        "server_label": "github",
        "server_url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {git_token}"},
        "allowed_tools": ["issue_write"],
    }

    try:
        logger.info("git_agent GIT calling response api")
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.GIT_PROMPT.format(
                github_url=github_url,
                sub_id=state["submission_id"],
                user_question=state["input"],
                initial_classification=state["mcp_output"],
            ),
            tools=[openai_mcp_tool],
        )
        logger.debug("git_agent response returned")

        # Extract GitHub issue URL from MCP call output
        state["github_issue"] = extract_mcp_output(
            resp, agent_name="git_agent", extract_url=True
        )
    except Exception as e:
        logger.info(f"git_agent Tool failed with error: '{e}'")

    agent_end_time = time.time()
    state["agent_timings"]["Git"] = agent_end_time - agent_start_time

    return state


def pod_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    tools_llm: "str | None",
) -> "WorkflowState":
    logger.info(f"K8S Agent request for submission: {state['submission_id']}")

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    state["active_agent"] = "Pod"

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    if not tools_llm:
        raise AgentRunMethodParameterError("tools_llm is required in git agent")

    openai_mcp_tool = {
        "type": "mcp",
        "server_label": "OpenShift / Kubernetes MCP Tools",
        "server_url": "http://localhost:8080/mcp",
        "require_approval": "never",
        "allowed_tools": ["pods_list_in_namespace"],
    }

    try:
        logger.debug(
            f"K8S Agent making MCP request for submission: {state['submission_id']}"
        )
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.POD_PROMPT.format(namespace=state["namespace"]),
            tools=[openai_mcp_tool],
        )
        logger.debug(
            f"K8S Agent successful return MCP request "
            f"for submission: {state['submission_id']}"
        )

        state["mcp_output"] = extract_mcp_output(resp, agent_name="pod_agent")
    except Exception as e:
        state["mcp_output"] = "K8s MCP Server not available"
        logger.info(
            f"K8s Agent unsuccessful return MCP request "
            f"for submission {state['submission_id']} with error: '{e}'"
        )

    agent_end_time = time.time()
    state["agent_timings"]["Pod"] = agent_end_time - agent_start_time

    return state


def perf_agent(
    state: "WorkflowState",
    openai_client: "OpenAI | None",
    tools_llm: "str | None",
) -> "WorkflowState":
    logger.info(f"K8S perf Agent request for submission: {state['submission_id']}")

    agent_start_time = time.time()
    if "agent_timings" not in state or state["agent_timings"] is None:
        state["agent_timings"] = {}
    state["active_agent"] = "Performance"

    # check if necessary variables exist
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    if not tools_llm:
        raise AgentRunMethodParameterError("tools_llm is required in git agent")

    openai_mcp_tool = {
        "type": "mcp",
        "server_label": "OpenShift / Kubernetes MCP Tools",
        "server_url": "http://localhost:8080/mcp",
        "require_approval": "never",
        "allowed_tools": ["pods_top"],
    }
    try:
        logger.debug(
            f"K8S perf Agent making MCP request "
            f"for submission: {state['submission_id']}"
        )
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.PERF_PROMPT.format(namespace=state["namespace"]),
            tools=[openai_mcp_tool],
        )
        logger.debug(
            f"K8S perf Agent successful return MCP request "
            f"for submission: {state['submission_id']}"
        )

        state["mcp_output"] = extract_mcp_output(resp, agent_name="perf_agent")
    except Exception as e:
        state["mcp_output"] = "K8s MCP server not available"
        logger.info(
            f"K8s perf Agent unsuccessful return MCP request "
            f"for submission {state['submission_id']} with error: '{e}'"
        )

    agent_end_time = time.time()
    state["agent_timings"]["Performance"] = agent_end_time - agent_start_time

    return state

from typing import Any

from openai import OpenAI

from src.exceptions import AgentRunMethodParameterError
from src.models import ClassificationModel, SupportClassificationModel
from src.types import WorkflowAgentPrompts, WorkflowState
from src.utils import extract_mcp_output, logger


def classification_agent(
    state: "WorkflowState", **kwargs: "dict[str, Any]"
) -> "WorkflowState":

    # check if necessary variables exist
    openai_client: "OpenAI" = kwargs.get("openai_client")
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in classification agent"
        )

    topic_llm = kwargs.get("topic_llm")
    if topic_llm is None:
        raise AgentRunMethodParameterError("model is required in classification agent")

    guardrail_model = kwargs.get("guardrail_model")
    if guardrail_model is None:
        raise AgentRunMethodParameterError(
            "guardrail_model is required in classification agent"
        )

    # checking if the input is safe
    safety_response = openai_client.moderations.create(
        model=guardrail_model, input=state.input
    )

    for moderation in safety_response.results:
        if not moderation.flagged:
            continue

        # TODO: decide log level
        logger.info(
            f"Classification result: '{state.input}' is flagged as '{moderation}'"
        )
        state.decision = "unsafe"
        state.data = state.input
        flagged_categories = [
            key
            for key, value in moderation.categories.model_extra.items()
            if value is True
        ]
        state.classification_message = f"Classification result: '{state.input}' is flagged for: {', '.join(flagged_categories)}"
        return state

    # NOTE: replaced langchain dependency with direct openai calls
    response = openai_client.responses.parse(
        model=topic_llm,
        input=WorkflowAgentPrompts.CLASIFICATION_PROMPT.format(state_input=state.input),
        text_format=ClassificationModel,
    )

    # NOTE: Using OpenAI directly for a structured output.
    # see https://platform.openai.com/docs/guides/structured-outputs#examples

    # check if parsed response has valid structure
    if not (
        response is not None
        and response.output_parsed is not None
        and (response.output_parsed.classification in ("legal", "support"))
    ):
        logger.error("Failed to get classification response from the model.")
        state.decision = "unknown"
        state.data = state.input
        state.classification_message = "Unable to determine request type."
        return state

    classification_result = response.output_parsed
    logger.info(
        f"Classification result: {classification_result} for input '{state.input}'"
    )

    state.decision = classification_result.classification
    state.data = state.input

    return state


def support_classification_agent(
    state: "WorkflowState", **kwargs: "dict[str, Any]"
) -> "WorkflowState":
    # check if necessary variables exist
    openai_client: "OpenAI" = kwargs.get("openai_client")
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support classification agent"
        )

    topic_llm = kwargs.get("topic_llm")
    if topic_llm is None:
        raise AgentRunMethodParameterError(
            "model is required in support classification agent"
        )

    # NOTE: replaced langchain dependency with direct openai calls
    response = openai_client.responses.parse(
        model=topic_llm,
        input=WorkflowAgentPrompts.SUPPORT_CLASIFICATION_PROMPT.format(
            state_input=state.input
        ),
        text_format=SupportClassificationModel,
    )
    classification_result = response.output_parsed

    # TODO: check parsing errors
    logger.info(
        f"Support Classification result: {classification_result} for input '{state.input}'"
        # f"and parsing error {parsing_error}"
    )
    state.namespace = classification_result.namespace
    state.decision = (
        classification_result.classification
        if classification_result.performance in ("true", "performance issue")
        else "perf"
    )
    state.data = state.input
    # TODO: This should be moved to agent level
    # sub_id = state["submissionID"]
    # saved_state = submission_states.get(sub_id, {})
    # state["classification_message"] = saved_state.get(
    #     "classification_message", state.get("classification_message", "")
    # )
    # submission_states[sub_id] = state
    return state


def git_agent(state: "WorkflowState", **kwargs: "dict[str, Any]") -> "WorkflowState":
    logger.debug(f"git Agent request for submission: {state.submission_id}")

    # check if necessary variables exist
    openai_client: "OpenAI" = kwargs.get("openai_client")
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    git_token: "str" = kwargs.get("git_token")
    if not git_token:
        raise AgentRunMethodParameterError("git_token is required in git agent")

    tools_llm: "str" = kwargs.get("tools_llm")
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
                github_url=kwargs.get("github_url", "unknown"),
                sub_id=state.submission_id,
                user_question=state.input,
                initial_classification=state.mcp_output,
            ),
            tools=[openai_mcp_tool],
        )
        logger.debug("git_agent response returned")

        # Extract GitHub issue URL from MCP call output
        state.github_issue = extract_mcp_output(
            resp, agent_name="git_agent", extract_url=True
        )
    except Exception as e:
        logger.info(f"git_agent Tool failed with error: '{e}'")
    return state


def pod_agent(state: "WorkflowState", **kwargs: "dict[str, Any]") -> "WorkflowState":
    logger.info(f"K8S Agent request for submission: {state.submission_id}")

    # check if necessary variables exist
    openai_client: "OpenAI" = kwargs.get("openai_client")
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    tools_llm: "str" = kwargs.get("tools_llm")
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
            f"K8S Agent making MCP request for submission: {state.submission_id}"
        )
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.POD_PROMPT.format(namespace=state.namespace),
            tools=[openai_mcp_tool],
        )
        logger.debug(
            f"K8S Agent successful return MCP request for submission: {state.submission_id}"
        )

        state.mcp_output = extract_mcp_output(resp, agent_name="pod_agent")
    except Exception as e:
        logger.info(
            f"K8s Agent unsuccessful return MCP request for submission {state.submission_id} with error: '{e}'"
        )
    return state


def perf_agent(state: "WorkflowState", **kwargs: "dict[str]") -> "WorkflowState":
    logger.info(f"K8S perf Agent request for submission: {state.submission_id}")

    # check if necessary variables exist
    openai_client: "OpenAI" = kwargs.get("openai_client")
    if openai_client is None:
        raise AgentRunMethodParameterError(
            "openai_client is required in support git agent"
        )

    tools_llm: "str" = kwargs.get("tools_llm")
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
            f"K8S perf Agent making MCP request for submission: {state['submissionID']}"
        )
        resp = openai_client.responses.create(
            model=tools_llm,
            input=WorkflowAgentPrompts.PERF_PROMPT.format(namespace=state.namespace),
            tools=[openai_mcp_tool],
        )
        logger.debug(
            f"K8S perf Agent successful return MCP request for submission: {state['submissionID']}"
        )

        state.mcp_output = extract_mcp_output(resp, agent_name="perf_agent")
    except Exception as e:
        logger.info(
            f"K8s perf Agent unsuccessful return MCP request for submission {state['submissionID']} with error: '{e}'"
        )
    return state

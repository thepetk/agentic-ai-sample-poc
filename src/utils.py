import logging
import os
import json
from typing_extensions import Literal
from llama_stack_client.types import ResponseObject

from src.types import WorkflowState


log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)


def clean_text(text: "str") -> "str":
    """
    cleans text to handle encoding issues.
    """
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("ascii", "ignore").decode("ascii")


def route_to_next_node(
    state: "WorkflowState",
) -> "Literal['legal_agent', 'support_agent', '__end__']":
    if state["decision"] == "legal":
        return "legal_agent"
    elif state["decision"] == "support":
        return "support_agent"
    else:
        return "__end__"


def support_route_to_next_node(
    state: "WorkflowState",
) -> "Literal['pod_agent', 'perf_agent', 'git_agent', '__end__']":
    if state["decision"] == "pod":
        return "pod_agent"
    elif state["decision"] == "git":
        return "git_agent"
    elif state["decision"] == "perf":
        return "perf_agent"

    return "__end__"


def extract_rag_response_text(rag_response: "ResponseObject") -> "str":
    """
    extracts text content from RAG response output.
    """
    _res_text = ""
    for output_item in rag_response.output:
        if not hasattr(output_item, "type"):
            continue

        if output_item.type in ("text", "message"):
            if hasattr(output_item, "content") and isinstance(
                output_item.content, list
            ):
                for content in output_item.content:
                    if not hasattr(content, "text"):
                        continue

                    _res_text += content.text + "\n"

            elif hasattr(output_item, "text"):
                _res_text += output_item.text + "\n"

        elif output_item.type == "file_search_call":
            logger.debug(
                f"RAG file_search executed with queries: {getattr(output_item, 'queries', [])}"
            )

    return _res_text.strip()


def extract_mcp_output(
    response: "ResponseObject", agent_name: "str" = "agent", extract_url: "bool" = False
) -> "str":
    """
    extracts MCP call output from a response object.
    """
    mcp_output = ""

    for item in response.output:
        item_type = item.__class__.__name__

        if item_type not in ("McpCall", "ResponseOutputMessage"):
            logger.debug(f"{agent_name}: Unexpected output item type: {item_type}")
            continue

        if item_type == "McpCall":
            if extract_url:
                # case: git agent - extract URL from JSON output
                try:
                    output_json = json.loads(item.output)
                    mcp_output = output_json.get("url", item.output)
                    logger.info(f"{agent_name}: GitHub issue created: {mcp_output}")
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"{agent_name}: Failed to parse MCP output: {e}")
                    mcp_output = item.output if hasattr(item, "output") else ""
            else:
                # case: other agents - return raw output
                mcp_output = item.output
                logger.info(f"{agent_name}: MCP call completed")
                logger.debug(f"{agent_name}: MCP output: {item.output}")

            break

        else:
            if hasattr(item, "content") and len(item.content) > 0:
                logger.debug(f"{agent_name} response message: {item.content[0].text}")

    return mcp_output

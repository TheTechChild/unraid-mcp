"""Docker container management.

Provides the `unraid_docker` tool with 15 actions for container lifecycle,
logs, networks, and update management.
"""

import re
from typing import Any, Literal

from fastmcp import FastMCP

from ..config.logging import logger
from ..core.client import make_graphql_request
from ..core.exceptions import ToolError


QUERIES: dict[str, str] = {
    "list": """
        query ListDockerContainers {
          docker { containers(skipCache: false) {
            id names image state status autoStart
          } }
        }
    """,
    "details": """
        query GetContainerDetails {
          docker { containers(skipCache: false) {
            id names image imageId command created
            ports { ip privatePort publicPort type }
            sizeRootFs labels state status
            hostConfig { networkMode }
            networkSettings mounts autoStart
          } }
        }
    """,
    "logs": """
        query GetContainerLogs($id: PrefixedID!, $tail: Int) {
          docker { logs(id: $id, tail: $tail) }
        }
    """,
    "networks": """
        query GetDockerNetworks {
          docker { networks { id name driver scope } }
        }
    """,
}

MUTATIONS: dict[str, str] = {
    "start": """
        mutation StartContainer($id: PrefixedID!) {
          docker { start(id: $id) { id names state status } }
        }
    """,
    "stop": """
        mutation StopContainer($id: PrefixedID!) {
          docker { stop(id: $id) { id names state status } }
        }
    """,
}

DESTRUCTIVE_ACTIONS: set[str] = set()
_ACTIONS_REQUIRING_CONTAINER_ID = {
    "start",
    "stop",
    "restart",
    "details",
    "logs",
}
# Actions not available in Unraid API v4.29.2 GraphQL schema
_UNAVAILABLE_ACTIONS = {"pause", "unpause", "remove", "update", "update_all", "port_conflicts", "check_updates", "network_details"}
ALL_ACTIONS = set(QUERIES) | set(MUTATIONS) | {"restart", "logs"} | _UNAVAILABLE_ACTIONS

DOCKER_ACTIONS = Literal[
    "list",
    "details",
    "start",
    "stop",
    "restart",
    "pause",
    "unpause",
    "remove",
    "update",
    "update_all",
    "logs",
    "networks",
    "network_details",
    "port_conflicts",
    "check_updates",
]

# Docker container IDs: 64 hex chars + optional suffix (e.g., ":local")
_DOCKER_ID_PATTERN = re.compile(r"^[a-f0-9]{64}(:[a-z0-9]+)?$", re.IGNORECASE)


def _safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys, handling None intermediates."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def find_container_by_identifier(
    identifier: str, containers: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find a container by ID or name with fuzzy matching.

    Match priority:
      1. Exact ID match
      2. Exact name match (case-sensitive)
      3. Name starts with identifier (case-insensitive)
      4. Name contains identifier as substring (case-insensitive)

    Note: Short identifiers (e.g. "db") may match unintended containers
    via substring. Use more specific names or IDs for precision.
    """
    if not containers:
        return None

    # Priority 1 & 2: exact matches
    for c in containers:
        if c.get("id") == identifier:
            return c
        if identifier in c.get("names", []):
            return c

    id_lower = identifier.lower()

    # Priority 3: prefix match (more precise than substring)
    for c in containers:
        for name in c.get("names", []):
            if name.lower().startswith(id_lower):
                logger.info(f"Prefix match: '{identifier}' -> '{name}'")
                return c

    # Priority 4: substring match (least precise)
    for c in containers:
        for name in c.get("names", []):
            if id_lower in name.lower():
                logger.info(f"Substring match: '{identifier}' -> '{name}'")
                return c

    return None


def get_available_container_names(containers: list[dict[str, Any]]) -> list[str]:
    """Extract all container names for error messages."""
    names: list[str] = []
    for c in containers:
        names.extend(c.get("names", []))
    return names


async def _resolve_container_id(container_id: str) -> str:
    """Resolve a container name/identifier to its actual PrefixedID."""
    if _DOCKER_ID_PATTERN.match(container_id):
        return container_id

    logger.info(f"Resolving container identifier '{container_id}'")
    list_query = """
        query ResolveContainerID {
          docker { containers(skipCache: true) { id names } }
        }
    """
    data = await make_graphql_request(list_query)
    containers = _safe_get(data, "docker", "containers", default=[])
    resolved = find_container_by_identifier(container_id, containers)
    if resolved:
        actual_id = str(resolved.get("id", ""))
        logger.info(f"Resolved '{container_id}' -> '{actual_id}'")
        return actual_id

    available = get_available_container_names(containers)
    msg = f"Container '{container_id}' not found."
    if available:
        msg += f" Available: {', '.join(available[:10])}"
    raise ToolError(msg)


def register_docker_tool(mcp: FastMCP) -> None:
    """Register the unraid_docker tool with the FastMCP instance."""

    @mcp.tool()
    async def unraid_docker(
        action: DOCKER_ACTIONS,
        container_id: str | None = None,
        network_id: str | None = None,
        *,
        confirm: bool = False,
        tail_lines: int = 100,
    ) -> dict[str, Any]:
        """Manage Docker containers, networks, and updates.

        Actions:
          list - List all containers
          details - Detailed info for a container (requires container_id)
          start - Start a container (requires container_id)
          stop - Stop a container (requires container_id)
          restart - Stop then start a container (requires container_id)
          pause - Pause a container (requires container_id)
          unpause - Unpause a container (requires container_id)
          remove - Remove a container (requires container_id, confirm=True)
          update - Update a container to latest image (requires container_id)
          update_all - Update all containers with available updates
          logs - Get container logs (requires container_id, optional tail_lines)
          networks - List Docker networks
          network_details - Details of a network (requires network_id)
          port_conflicts - Check for port conflicts
          check_updates - Check which containers have updates available
        """
        if action not in ALL_ACTIONS:
            raise ToolError(f"Invalid action '{action}'. Must be one of: {sorted(ALL_ACTIONS)}")

        if action in DESTRUCTIVE_ACTIONS and not confirm:
            raise ToolError(f"Action '{action}' is destructive. Set confirm=True to proceed.")

        if action in _ACTIONS_REQUIRING_CONTAINER_ID and not container_id:
            raise ToolError(f"container_id is required for '{action}' action")

        if action == "network_details" and not network_id:
            raise ToolError("network_id is required for 'network_details' action")

        try:
            logger.info(f"Executing unraid_docker action={action}")

            # Actions not available in Unraid API v4.29.2
            if action in _UNAVAILABLE_ACTIONS:
                raise ToolError(
                    f"Action '{action}' is not available in the Unraid GraphQL API. "
                    "Use the unraid-docker MCP server (Docker over SSH) for this operation."
                )

            # --- Read-only queries ---
            if action == "list":
                data = await make_graphql_request(QUERIES["list"])
                containers = _safe_get(data, "docker", "containers", default=[])
                return {"containers": list(containers) if isinstance(containers, list) else []}

            if action == "details":
                data = await make_graphql_request(QUERIES["details"])
                containers = _safe_get(data, "docker", "containers", default=[])
                container = find_container_by_identifier(container_id or "", containers)
                if container:
                    return container
                available = get_available_container_names(containers)
                msg = f"Container '{container_id}' not found."
                if available:
                    msg += f" Available: {', '.join(available[:10])}"
                raise ToolError(msg)

            if action == "logs":
                raise ToolError(
                    "Container logs are not available via the Unraid GraphQL API. "
                    "Use the unraid-docker MCP server (Docker over SSH) for logs."
                )

            if action == "networks":
                data = await make_graphql_request(QUERIES["networks"])
                networks = _safe_get(data, "docker", "networks", default=[])
                return {"networks": list(networks) if isinstance(networks, list) else []}

            # --- Mutations ---
            if action == "restart":
                actual_id = await _resolve_container_id(container_id or "")
                stop_data = await make_graphql_request(
                    MUTATIONS["stop"],
                    {"id": actual_id},
                    operation_context={"operation": "stop"},
                )
                stop_was_idempotent = stop_data.get("idempotent_success", False)
                start_data = await make_graphql_request(
                    MUTATIONS["start"],
                    {"id": actual_id},
                    operation_context={"operation": "start"},
                )
                if start_data.get("idempotent_success"):
                    result = {}
                else:
                    result = _safe_get(start_data, "docker", "start", default={})
                response: dict[str, Any] = {
                    "success": True,
                    "action": "restart",
                    "container": result,
                }
                if stop_was_idempotent:
                    response["note"] = "Container was already stopped before restart"
                return response

            # Single-container mutations (start, stop)
            if action in MUTATIONS:
                actual_id = await _resolve_container_id(container_id or "")
                op_context: dict[str, str] | None = {"operation": action}
                data = await make_graphql_request(
                    MUTATIONS[action],
                    {"id": actual_id},
                    operation_context=op_context,
                )

                if data.get("idempotent_success"):
                    return {
                        "success": True,
                        "action": action,
                        "idempotent": True,
                        "message": f"Container already in desired state for '{action}'",
                    }

                docker_data = data.get("docker") or {}
                result = docker_data.get(action)
                return {
                    "success": True,
                    "action": action,
                    "container": result,
                }

            raise ToolError(f"Unhandled action '{action}' -- this is a bug")

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Error in unraid_docker action={action}: {e}", exc_info=True)
            raise ToolError(f"Failed to execute docker/{action}: {e!s}") from e
    logger.info("Docker tool registered successfully")

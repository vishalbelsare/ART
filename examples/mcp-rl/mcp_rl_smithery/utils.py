from contextlib import asynccontextmanager

import mcp.types as types
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


@asynccontextmanager
async def mcp_session(smithery_mcp_url: str):
    """
    Connects to the remote Smithery MCP server using the full URL that includes
    your API key & profile. No OAuth provider is used.
    """
    async with streamablehttp_client(smithery_mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools_and_resources(smithery_mcp_url: str):
    """Return (tools_result, resources_result) from the remote Smithery server."""
    async with mcp_session(smithery_mcp_url) as session:
        tools = await session.list_tools()
        try:
            resources = await session.list_resources()
        except Exception:
            # Some servers don't implement resources; keep interface stable
            class _Empty:
                resources = []

            resources = _Empty()
        return tools, resources


async def call_mcp_tool(smithery_mcp_url: str, tool_name: str, arguments: dict):
    """Invoke a tool on the remote Smithery server and return the CallToolResult."""
    async with mcp_session(smithery_mcp_url) as session:
        return await session.call_tool(tool_name, arguments)


def get_content_text(result: types.CallToolResult) -> str:
    # Extract text content from MCP result
    if hasattr(result, "content") and result.content:
        if isinstance(result.content, list):
            # Handle list of content items
            content_text = ""
            for item in result.content:
                if isinstance(item, types.TextContent):
                    content_text += item.text
                else:
                    content_text += str(item)
        elif isinstance(result.content[0], types.TextContent):
            content_text = result.content[0].text
        else:
            content_text = str(result.content)
    else:
        content_text = str(result)

    return content_text

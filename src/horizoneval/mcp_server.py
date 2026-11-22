"""horizoneval as an MCP server: the tools of horizoneval.tools, for any assistant that speaks MCP.

    uv run --extra mcp horizoneval-mcp            # stdio; add it to Claude Desktop, Claude Code, Cursor, ...

Each tool returns the same canonical JSON document as the Python API and the command line, both as text and as
structured content. Install with the `mcp` extra.
"""

import anyio

from horizoneval import tools


def build_server():
    import mcp_types as types
    from mcp.server import Server

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(name=d["name"], description=d["description"],
                                                       input_schema=d["input_schema"])
                                            for d in tools.definitions("mcp")])

    async def call_tool(ctx, params):
        try:
            doc = await anyio.to_thread.run_sync(lambda: tools.call(params.name, params.arguments or {}))
        except Exception as e:  # the assistant sees the reason and can fix its call
            return types.CallToolResult(content=[types.TextContent(type="text", text=f"{type(e).__name__}: {e}")],
                                        is_error=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=tools.dumps(doc))],
                                    structured_content=doc)

    from horizoneval.result import RESULT_VERSION

    return Server("horizoneval", version=RESULT_VERSION,
                  instructions="Evaluate time series forecasts and help choose one. Start with `check`, then `score`; "
                               "lead with the result's verdict, in plain language.",
                  on_list_tools=list_tools, on_call_tool=call_tool)


def main():
    from mcp.server.stdio import stdio_server

    server = build_server()

    async def run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()

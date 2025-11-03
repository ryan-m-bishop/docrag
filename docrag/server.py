import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from typing import Any
import logging

from .config import ConfigManager
from .embeddings import EmbeddingGenerator
from .vectordb import VectorDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docrag-server")


class DocRAGServer:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.embeddings = EmbeddingGenerator()
        self.vectordb = VectorDB(self.config_manager.vectordb_dir)
        self.server = Server("docrag-server")

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            collections = self.vectordb.list_collections()
            return [
                Tool(
                    name="search_docs",
                    description=(
                        "Search through indexed documentation collections. "
                        "Returns relevant documentation chunks that match the query. "
                        "Use this when you need to find specific information in technical documentation."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query - describe what documentation you're looking for"
                            },
                            "collection": {
                                "type": "string",
                                "description": f"Optional: specific collection to search. Available: {', '.join(collections) if collections else 'none'}. If not specified, searches all active collections.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="list_collections",
                    description="List all available documentation collections that can be searched.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            if name == "search_docs":
                return await self._search_docs(arguments)
            elif name == "list_collections":
                return await self._list_collections()
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_docs(self, args: dict) -> list[TextContent]:
        query = args["query"]
        collection = args.get("collection")
        limit = args.get("limit", 5)

        logger.info(f"Searching for: {query} in collection: {collection or 'all'}")

        # Generate query embedding
        query_embedding = self.embeddings.embed_single(query)

        # Determine which collections to search
        if collection:
            collections = [collection]
        else:
            config = self.config_manager.load_config()
            collections = config.active_collections or self.vectordb.list_collections()

        # Search each collection
        all_results = []
        for coll in collections:
            results = self.vectordb.search(coll, query_embedding.tolist(), limit)
            for result in results:
                result['collection'] = coll
            all_results.extend(results)

        # Sort by score and limit
        all_results.sort(key=lambda x: x.get('_distance', float('inf')))
        all_results = all_results[:limit]

        if not all_results:
            return [TextContent(
                type="text",
                text=f"No results found for query: {query}"
            )]

        # Format results
        formatted_results = self._format_results(all_results, query)

        return [TextContent(type="text", text=formatted_results)]

    async def _list_collections(self) -> list[TextContent]:
        collections = self.vectordb.list_collections()

        if not collections:
            return [TextContent(
                type="text",
                text="No collections available. Use 'docrag add <name>' to create a collection."
            )]

        config = self.config_manager.load_config()
        active = config.active_collections

        result = "Available documentation collections:\n\n"
        for coll in collections:
            status = "active" if coll in active else "inactive"
            result += f"- {coll} ({status})\n"

        return [TextContent(type="text", text=result)]

    def _format_results(self, results: list, query: str) -> str:
        """Format search results for display"""
        output = f"# Documentation Search Results\n\n"
        output += f"**Query:** {query}\n"
        output += f"**Found:** {len(results)} relevant sections\n\n"
        output += "---\n\n"

        for i, result in enumerate(results, 1):
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            collection = result.get('collection', 'unknown')
            source = metadata.get('source', 'unknown')
            score = result.get('_distance', 0)

            output += f"## Result {i}\n"
            output += f"**Collection:** {collection}\n"
            output += f"**Source:** {source}\n"
            output += f"**Relevance Score:** {1 - score:.3f}\n\n"
            output += f"{text}\n\n"
            output += "---\n\n"

        return output

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting DocRAG MCP Server...")

        # Ensure config is initialized
        self.config_manager.init()

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    server = DocRAGServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())

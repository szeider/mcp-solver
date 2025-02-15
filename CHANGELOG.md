# Changelog

All notable changes to this project are documented here.

### [2.2.0] - 2025-02-15

- **New Feature:** Integrated static prompt endpoints (`prompts/list` and `prompts/get`) to advertise MCP prompt templates ("quick_prompt" and "detailed_prompt") without requiring any arguments.
- **New Feature:** Advertised detailed tool capabilities by adding descriptive metadata for each tool in the server’s capabilities declaration.
- **Improvement:** Enhanced error reporting for tool endpoints with improved logging and standardized error responses.
- **Update:** Refactored server initialization to explicitly log the declared capabilities for greater transparency and easier debugging.

### [2.1.0] - 2025-02-09

- **Update:** Change minimum Python requirement to 3.11+ (to support `asyncio.timeout`).
- **Update:** Bump dependency on `mcp` to version 1.2.0 or later.
- **Improvement:** Update tool handler messages so that "delete_item" and "replace_item" commands correctly report the operation performed.
- **Update:** Miscellaneous documentation and cleanup.

### [2.0.0] - 2024-12-29

- Major change: Use item-based editing.

### [1.0.0] - 2024-12-21

- Major change: Use line-based model editing.
- Makes parameter handling obsolete.
- Added dynamic knowledge base handling.

### [0.2.1] - 2024-12-16

- Changed parameter handling.

### [0.2.0] - 2024-12-15

- Initial release.
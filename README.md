# Eventor API Client

A Python client for integrating with the Eventor orienteering API, providing access to events, entries, and results.

## Features

- Access to event results, entries, and organization data
- Configurable caching system for reduced API calls
- MCP (Model Context Protocol) server integration for AI assistants
- Exposes 3 tools for MCP server:
  - get_weekly_club_results() Fetches weekly club results can be used to have the LLM to summarize the results and create a report
  - get_personal_events() Fetches personal events for a given personId (config file) and given period of time
  - get_detailed_result_for_event_and_person() Fetches detailed results for a specific event and personId (config file)
- It does some parsing and tries to provide the data in a LLM friendly format
- Debug mode to run the client standalone and generate JSON output for review
- Configurable API key and personId for secure access


## Installation

1. Clone the repository:
```bash
git clone https://github.com/grapatin/eventor-mcp-server.git
cd eventor-mcp-server
```

2. Set up a virtual environment:
```bash
uv venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Configure your API key:
```bash
cp eventor_config_example.py eventor_config.py
# Edit eventor_config.py with your actual API key, personId, and other settings
```

## Usage

To debug just start it in debug mode, it has a __main__ function that will run the function stand alone and produce a json file with the results for review.

```bash
python -m eventor
```

### Install MCP server for Claude

Add this to cluade_desktopr_config.json

```json
    "Eventor Results API": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/eventor-mcp-server",
        "run",
        "mcp",
        "run",
        "mcp_server.py"
      ]
    }
```

## Configuration

Configuration is done via the `eventor_config.py` file. This file contains sensitive information and is excluded from version control via `.gitignore`.
Add both eventor private key (talk to org eventor admin) and your own private personId. (can be found in out put json file)


## License

[MIT License](LICENSE)

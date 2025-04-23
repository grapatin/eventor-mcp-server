#!/usr/bin/env python3
"""
MCP Server for Eventor API - Allows AI to access Eventor results data
"""
from typing import Literal, Optional, Union
from fastmcp import FastMCP
from eventor_api import eventor_org_result_summary
from eventor_api import eventor_get_recent_events_for_person
from eventor_api import event_get_detailed_result_for_event_and_person


# Create MCP server
mcp = FastMCP(
    name="Eventor Results API",
    instructions="MCP server for accessing Eventor results data",
)

@mcp.tool()
def get_weekly_club_results() -> Union[dict, Literal["error"]]:
    """
    Hämta veckans resultat för min klubb i eventor.
    Returns the results_all_relevant_events_org to the AI.
    """
    results = eventor_org_result_summary()
    
    return results

@mcp.tool()
def get_personal_events(daysback: int = 7) -> Union[dict, Literal["error"]]:
    """
    Hämta mina resultat i eventor.
    Returns the results_all_relevant_events_person to the AI.
    Inparameter är hur många dagar bakåt i tiden som vi skall leta (default 7).
    """
    results = eventor_get_recent_events_for_person(daysback=daysback)
    
    return results

@mcp.tool()
def get_detailed_result_for_event_and_person(event_id: str) -> Union[dict, Literal["error"]]:
    """
    Hämta detaljerat resultat för en specifik person och event.
    Returns a LLM-friendly JSON with metadata and result for the person of interest in the event.
    Inparameter är eventets ID (event_id).
    """
    try:
        result = event_get_detailed_result_for_event_and_person(event_id)
        if result is None:
            return "error"
        return result
    except Exception as e:
        return "error"

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(mcp, host="0.0.0.0", port=8000)
    #print(get_weekly_club_results())
    print(get_personal_events(daysback=14))
#!/usr/bin/env python3
"""
Eventor API Client
A simple client for integrating with the Eventor orienteering API
"""

import json
import urllib.parse
import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import requests
import xml.etree.ElementTree as ET


try:
    from eventor_config import API_KEY
    from eventor_config import PERSON_ID
except ImportError:
    print("Using API key from environment" if API_KEY else 
          "No API key found. Set EVENTOR_API_KEY or create eventor_config.py")

# API configuration
API_BASE_URL = "https://eventor.orientering.se/api"

# Cache configuration
CACHE_DIR = Path(os.path.expanduser("~/.eventor_cache"))
CACHE_EXPIRY = timedelta(hours=72)  # Default: Cache expires after 72 hours

# Try to load custom cache expiry from config
try:
    from eventor_config import CACHE_EXPIRY_HOURS
    CACHE_EXPIRY = timedelta(hours=CACHE_EXPIRY_HOURS)
except (ImportError, AttributeError):
    pass

# Define explicit exports for cleaner imports in other modules
__all__ = [
    'get_organisation', 
    'get_events_with_org_entries',
    'get_event_results',
    'get_organisation_results',
    'eventor_org_result_summary',
    'eventor_get_recent_events_for_person',
    'parse_results',
    'save_json',
    'clear_cache',
    'extract_splits'
]

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(url, params=None):
    """Generate a unique cache key based on URL and parameters"""
    key_parts = [url]
    if params:
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        key_parts.extend(f"{k}={v}" for k, v in sorted_params)
    
    # Create a hash for the key parts
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cache_path(cache_key):
    """Get the file path for a cache key"""
    return CACHE_DIR / f"{cache_key}.xml"

def save_to_cache(cache_key, data):
    """Save XML response data to the cache with metadata"""
    ensure_cache_dir()
    cache_path = get_cache_path(cache_key)
    
    # Save XML content
    with open(cache_path, "wb") as f:
        if isinstance(data, bytes):
            f.write(data)
        else:
            f.write(data.encode("utf-8"))
    
    # Save metadata in a separate file
    meta_path = CACHE_DIR / f"{cache_key}.meta"
    with open(meta_path, "w", encoding="utf-8") as f:
        metadata = {"timestamp": datetime.now().isoformat()}
        json.dump(metadata, f, ensure_ascii=False)

def get_from_cache(cache_key, max_age=None):
    """Get XML response data from the cache if it exists and is not expired"""
    if max_age is None:
        max_age = CACHE_EXPIRY
    
    # Check metadata for timestamp
    meta_path = CACHE_DIR / f"{cache_key}.meta"
    if not meta_path.exists():
        return None
        
    with open(meta_path, "r", encoding="utf-8") as f:
        try:
            metadata = json.load(f)
            timestamp = datetime.fromisoformat(metadata["timestamp"])
            
            # Check if cache is expired
            if datetime.now() - timestamp > max_age:
                return None
        except (json.JSONDecodeError, KeyError):
            return None
    
    # Get the actual XML data
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return f.read()
    
    return None

def clear_cache(older_than=None):
    """Clear the cache, optionally only items older than specified"""
    if not CACHE_DIR.exists():
        return
    
    for cache_file in CACHE_DIR.glob("*.*"):
        try:
            if older_than:
                # Check if it's a metadata file
                if cache_file.suffix == ".meta":
                    with open(cache_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                    if datetime.now() - timestamp > older_than:
                        cache_file.unlink()
                        # Also remove the corresponding data file
                        xml_file = CACHE_DIR / f"{cache_file.stem}.xml"
                        if xml_file.exists():
                            xml_file.unlink()
            else:
                # Remove all files
                cache_file.unlink()
        except Exception as e:
            print(f"Error clearing cache file {cache_file}: {e}")

def parse_hms(hms: str) -> timedelta:
    """'1:13:36'  ->  timedelta(hours=1, minutes=13, seconds=36)"""
    parts = list(map(int, hms.split(":")))
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        raise ValueError("Invalid time format")
    return timedelta(hours=h, minutes=m, seconds=s)

def diff_hms(t1: str, t2: str) -> timedelta:
    """Return absolute difference between two h:mm:ss strings."""
    return abs(parse_hms(t2) - parse_hms(t1))

def format_hms(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s  = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}"

def hms_to_seconds(time_str):
    """Convert a time string like "1:13:36" to seconds (integer)"""
    if not time_str:
        return 0
        
    try:
        parts = list(map(int, time_str.split(":")))
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = parts
            return m * 60 + s
        elif len(parts) == 1:
            return parts[0]
    except (ValueError, TypeError):
        pass
    return 0

def extract_splits(xml_root):
    """Extract split times from XML and calculate leg times for each competitor."""
    class_split_data = {}
    
    for class_result in xml_root.iterfind(".//ClassResult"):
        cls_name = class_result.findtext("./EventClass/Name")
        if not cls_name:
            continue
            
        # Dictionary to store all competitor split data for this class
        class_competitors = []
        course_controls = []  # List to track control codes in order
        
        # Extract control information if available
        course_element = class_result.find(".//Course")
        if course_element is not None:
            for control in course_element.findall(".//Control"):
                control_code = control.findtext("ControlCode")
                if control_code:
                    course_controls.append(control_code)
        
        # Process each competitor
        for person_result in class_result.iterfind("./PersonResult"):
            result_element = person_result.find("./Result")
            if result_element is None:
                continue
                
            status_elem = result_element.find("./CompetitorStatus")
            comp_status = status_elem.get("value") if status_elem is not None else "Unknown"

            # Only include competitors who have a status other than "DidNotStart"
            # This is important for the splits to be correct
            if comp_status == "DidNotStart":
                continue
            
            # Extract competitor info
            person_element = person_result.find("./Person")
            org_element = person_result.find("./Organisation")
            
            if person_element is None or org_element is None:
                continue
                
            full_name = " ".join(
                part.text.strip()
                for part in person_element.findall("./PersonName/*")
                if part.text
            )
            
            org_name = org_element.findtext("Name", "")
            org_id = org_element.findtext("OrganisationId")
            
            # Build competitor data structure
            competitor = {
                "fullName": full_name,
                "club": org_name,
                "orgId": org_id,
                "rawSplits": [],
                "legTimes": []
            }
            
            # Extract timing information
            split_times = {}
            start_time = None
            finish_time = None
            total_time = None
            
            # Get start time
            start_time_elem = result_element.find("./StartTime/Clock")
            if start_time_elem is not None and start_time_elem.text:
                start_time = start_time_elem.text
                
            # Get finish time and total time
            finish_time_elem = result_element.find("./FinishTime/Clock")
            if finish_time_elem is not None and finish_time_elem.text:
                finish_time = finish_time_elem.text
                
            # Get total time (important for calculating the last leg correctly)
            total_time_elem = result_element.findtext("Time")
            if total_time_elem:
                total_time = total_time_elem
            
            # Get all split times
            for split_elem in result_element.findall("./SplitTime"):
                seq = split_elem.get("sequence")
                ctrl_code = split_elem.findtext("ControlCode")
                split_time = split_elem.findtext("Time")
                
                if seq and ctrl_code and split_time:
                    split_times[int(seq)] = {
                        "controlCode": ctrl_code,
                        "time": split_time
                    }
            
            # Sort by sequence
            sorted_splits = [split_times[seq] for seq in sorted(split_times.keys())]
            competitor["rawSplits"] = sorted_splits
            
            # Create a combined list of all timing points in sequence (start, controls, finish)
            all_timing_points = []
            
            # Add start point if available
            if start_time:
                all_timing_points.append({
                    "point": "start",
                    "time": "0:00:00"  # Always start at zero for leg time calculations
                })
            
            # Add control points
            for split in sorted_splits:
                all_timing_points.append({
                    "point": split["controlCode"],
                    "time": split["time"]
                })
            
            # Add finish point if available
            if total_time:
                all_timing_points.append({
                    "point": "finish",
                    "time": total_time  # Use the total race time for finish
                })
            
            # Calculate leg times between consecutive timing points and running total
            leg_times = []
            running_total_td = timedelta(seconds=0)
            last_valid_total = timedelta(seconds=0)
            
            for i in range(len(all_timing_points) - 1):
                from_point = all_timing_points[i]
                to_point = all_timing_points[i + 1]
                
                try:
                    from_time = parse_hms(from_point["time"])
                    to_time = parse_hms(to_point["time"])
                    leg_time = to_time - from_time
                    
                    # Skip invalid splits (negative time or time going backwards)
                    if leg_time.total_seconds() < 0:
                        continue
                        
                    # Add leg time to running total for cumulative time
                    running_total_td += leg_time
                    
                    # Ensure running total is always increasing
                    if running_total_td < last_valid_total:
                        continue
                    
                    last_valid_total = running_total_td
                    leg_time_str = format_hms(leg_time)
                    running_total_str = format_hms(running_total_td)
                    
                    leg = {
                        "controls": f"{from_point['point']}-{to_point['point']}",
                        "time": leg_time_str,  # This is the leg time (not cumulative)
                        "legTime": leg_time_str,  # Duplicate for consistency
                        "runningTotal": running_total_str  # Cumulative time so far
                    }
                    leg_times.append(leg)
                except (ValueError, KeyError):
                    # Skip if we can't calculate the leg time
                    continue
            
            competitor["legTimes"] = leg_times
            class_competitors.append(competitor)
        
        class_split_data[cls_name] = class_competitors
    
    # Now we have all the raw data, compute positions and time-behind
    processed_split_data = {}
    
    for cls_name, competitors in class_split_data.items():
        # If class has no competitors with splits, skip
        if not competitors:
            continue
            
        # Group all leg times by control pair (e.g., "31-32")
        control_leg_times = {}
        
        for competitor in competitors:
            for leg in competitor["legTimes"]:
                control_pair = leg["controls"]
                leg_time_str = leg.get("time")  # Use the leg time
                
                if not leg_time_str:
                    continue
                    
                if control_pair not in control_leg_times:
                    control_leg_times[control_pair] = []
                
                control_leg_times[control_pair].append({
                    "competitor": competitor["fullName"],
                    "club": competitor["club"],
                    "orgId": competitor["orgId"],
                    "legTime": leg_time_str,
                    "runningTotal": leg["runningTotal"]
                })
        
        # Calculate best time and position for each control leg
        for control_pair, times in control_leg_times.items():
            # Sort by leg time
            try:
                # First, parse all leg times to timedeltas for accurate comparison
                for result in times:
                    result["parsed_time"] = parse_hms(result["legTime"])
                
                # Sort by the parsed timedelta
                sorted_times = sorted(times, key=lambda x: x["parsed_time"])
                
                # Calculate position with tie handling
                if sorted_times:
                    current_position = 1
                    previous_time = None
                    position_count = 0
                    
                    for i, result in enumerate(sorted_times):
                        current_time = result["parsed_time"]
                        
                        if previous_time is not None and current_time != previous_time:
                            # Different time than previous runner - assign position based on count
                            current_position = i + 1
                        
                        result["position"] = current_position
                        previous_time = current_time
                        
                        # The first position (lowest time) always has time_behind = "0:00:00"
                        if current_position == 1:
                            result["time_behind"] = "0:00:00"
                        else:
                            time_behind = result["parsed_time"] - sorted_times[0]["parsed_time"]
                            result["time_behind"] = format_hms(time_behind)
                    
                    # Remove the temporary parsed_time field
                    for result in sorted_times:
                        del result["parsed_time"]
                            
                control_leg_times[control_pair] = sorted_times
            except ValueError:
                # Skip this control pair if we can't sort the times
                continue
        
        # Now update each competitor with their position and time behind
        for competitor in competitors:
            enhanced_leg_times = []
            compact_splits = []
            
            for leg in competitor["legTimes"]:
                control_pair = leg["controls"]
                
                if control_pair in control_leg_times:
                    # Find this competitor's result for this leg
                    leg_results = [
                        r for r in control_leg_times[control_pair] 
                        if r["competitor"] == competitor["fullName"]
                    ]
                    
                    if leg_results:
                        leg_result = leg_results[0]
                        # Create the verbose version for backward compatibility
                        enhanced_leg = {
                            "controls": control_pair,
                            "time": leg["time"],
                            "position": leg_result["position"],
                            "time_behind": leg_result["time_behind"],
                            "runningTotal": leg_result["runningTotal"]
                        }
                        enhanced_leg_times.append(enhanced_leg)
                        
                        # Create the compact version following our schema: ["ctrl", "tSec", "pos", "behSec", "totSec"]
                        try:
                            # Convert times to seconds
                            time_sec = int(parse_hms(leg["time"]).total_seconds())
                            behind_sec = int(parse_hms(leg_result["time_behind"]).total_seconds())
                            total_sec = int(parse_hms(leg_result["runningTotal"]).total_seconds())
                            
                            compact_leg = [
                                control_pair,           # ctrl
                                time_sec,               # tSec
                                leg_result["position"], # pos
                                behind_sec,             # behSec
                                total_sec               # totSec
                            ]
                            compact_splits.append(compact_leg)
                        except Exception:
                            # Skip if we can't convert the times
                            pass
            
            competitor["splits"] = enhanced_leg_times  # Keep backward compatibility
            competitor["compactSplits"] = compact_splits  # Add the new compact format
        
        processed_split_data[cls_name] = competitors
    
    return processed_split_data

def parse_results(xml_source: str | Path) -> dict[str, list[dict]]:
    """Return {className: [ {place, time, behind, fullName, club}, … ]}."""
    if isinstance(xml_source, str) and not xml_source.strip().startswith("<"):
        # This is a file path, not XML content
        with open(xml_source, "rb") as f:
            root = ET.fromstring(f.read())
    else:
        root = ET.fromstring(xml_source)

    data: dict[str, list[dict]] = {}
    for class_result in root.iterfind(".//ClassResult"):
        cls_name = class_result.findtext("./EventClass/Name")
        results = []
        position = 1
        
        for pr in class_result.iterfind("./PersonResult"):
            person = pr.find("./Person")
            org = pr.find("./Organisation")
            res = pr.find("./Result")
            
            status_elem = res.find("./CompetitorStatus") if res is not None else None
            comp_status = status_elem.get("value") if status_elem is not None else "Unknown"
            
            if comp_status == "DidNotStart":
                continue
                
            org_id = org.findtext("OrganisationId") if org is not None else None
            if position == 1:
                winnerTime = res.findtext("Time") if res is not None else None
            time = res.findtext("Time") if res is not None else None

            delta = diff_hms(winnerTime, time) if winnerTime and time else None
            behind = format_hms(delta) if delta else None

            full_name = " ".join(
                part.text.strip()
                for part in person.findall("./PersonName/*")
                if part.text
            )
            # Add personId to the result
            person_id = person.findtext("PersonId") if person is not None else None

            result = {
                "fullName": full_name,
                "personId": person_id,
                "club": org.findtext("Name") if org is not None else "",
                "orgId": org_id,
                "status": comp_status,
                "time": time,
            }
            
            if comp_status == "OK":
                result["place"] = position
                result["behind"] = behind
            else:
                result["place"] = None
                result["behind"] = None
                
            results.append(result)
            position += 1

        data[cls_name] = results

    return data

def get_headers():
    """Return the headers required for all API requests"""
    return {"ApiKey": API_KEY}

def get_organisation():
    """Get organisation information"""
    url = f"{API_BASE_URL}/organisation/apiKey"
    cache_key = get_cache_key(url)
    
    cached_data = get_from_cache(cache_key)
    if cached_data:
        print(f"Using cached organisation data")
        # Parse the XML data from cache
        root = ET.fromstring(cached_data)
        organisation = {child.tag: child.text for child in root}
        return organisation
    
    response = requests.get(url, headers=get_headers())
    if response.status_code == 200:
        save_to_cache(cache_key, response.content)
        root = ET.fromstring(response.content)
        organisation = {child.tag: child.text for child in root}
        return organisation
    return None

def get_events_with_org_entries(org_id=230, from_date=None, to_date=None):
    """Get competition entries for an organisation"""
    from_date = from_date or datetime.now().strftime("%Y-%m-%d 00:00:00")
    to_date = to_date or datetime.now().strftime("%Y-%m-%d 23:59:59")
    
    params = {
        "organisationIds": org_id,
        "fromEventDate": from_date,
        "toEventDate": to_date,
        "includeEventElement": "true",
        "includePersonElement": "true"
    }
    
    url = f"{API_BASE_URL}/entries"
    cache_key = f"raw_{get_cache_key(url, params)}"
    
    cached_xml = get_from_cache(cache_key)
    if cached_xml:
        print(f"Using cached API data for org {org_id} entries")
        xml_content = cached_xml
    else:
        url_with_params = f"{url}?{urllib.parse.urlencode(params)}"
        response = requests.get(url_with_params, headers=get_headers())
        if response.status_code == 200:
            xml_content = response.content
            save_to_cache(cache_key, xml_content)
        else:
            return None
    
    if xml_content:
        root = ET.fromstring(xml_content)
        event_info = {}
        
        for entry in root.findall("./Entry"):
            event_element = entry.find("./Event")
            if event_element is not None:
                event_id = event_element.findtext("EventId")
                event_name = event_element.findtext("Name")
                
                formatted_date = ""
                start_date_elem = event_element.find("StartDate")
                if start_date_elem is not None:
                    date_str = start_date_elem.findtext("Date")
                    if date_str:
                        try:
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                            formatted_date = dt.strftime("%Y%m%d")
                        except ValueError:
                            formatted_date = ""
                
                if event_id and event_id not in event_info:
                    event_info[event_id] = {
                        "eventName": event_name or f"Event {event_id}",
                        "date": formatted_date
                    }
        
        print("unique EventId count:", len(event_info))
        return event_info
    
    return None

def get_event_results(event_id, include_split_times=True, max_cache_age=None):
    """Get results for a specific event"""
    params = {
        "eventId": event_id,
        "includeSplitTimes": str(include_split_times).lower()
    }
    
    url = f"{API_BASE_URL}/results/event"
    cache_key = f"raw_{get_cache_key(url, params)}"
    
    cached_xml = get_from_cache(cache_key, max_age=max_cache_age)
    if cached_xml:
        print(f"Using cached API data for event {event_id}")
        xml_content = cached_xml
    else:
        url_with_params = f"{url}?{urllib.parse.urlencode(params)}"
        response = requests.get(url_with_params, headers=get_headers())
        if response.status_code == 200:
            xml_content = response.content
            save_to_cache(cache_key, xml_content)
        else:
            return None
    
    if xml_content:
        results = parse_results(xml_content)
        # Parse split times if requested
        if include_split_times:
            root = ET.fromstring(xml_content)
            split_data = extract_splits(root)
            # Add split data to results
            return {
                "results": results,
                "splits": split_data
            }
        return results
    
    return None

def get_organisation_results(org_id, event_id, include_split_times=True, max_cache_age=None):
    """Get results for an organisation in a specific event"""
    params = {
        "organisationIds": org_id,
        "eventId": event_id,
        "includeSplitTimes": str(include_split_times).lower()
    }
    
    url = f"{API_BASE_URL}/results/organisation"
    cache_key = f"raw_{get_cache_key(url, params)}"
    
    cached_data = get_from_cache(cache_key, max_age=max_cache_age)
    if cached_data:
        print(f"Using cached API data for org {org_id} in event {event_id}")
        # Parse XML data from cache
        xml_content = cached_data
        root = ET.fromstring(xml_content)
        return parse_results(root)
    else:
        url_with_params = f"{url}?{urllib.parse.urlencode(params)}"
        response = requests.get(url_with_params, headers=get_headers())
        if response.status_code == 200:
            xml_content = response.content
            save_to_cache(cache_key, xml_content)
            root = ET.fromstring(xml_content)
            return parse_results(root)
    
    return None

def save_json(data, filename):
    """Save data as JSON to a file in the json/ directory."""
    import os
    json_dir = os.path.join(os.path.dirname(__file__), 'json')
    os.makedirs(json_dir, exist_ok=True)
    filepath = os.path.join(json_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_slug(text):
    """
    Create a URL-friendly slug from text.
    Converts to lowercase, removes special characters, replaces spaces with hyphens.
    Also transliterates Swedish characters ä, å, ö to a and o.
    """
    if not text:
        return ""
        
    # Transliterate Swedish characters
    text = text.replace("å", "a").replace("ä", "a").replace("ö", "o")
    text = text.replace("Å", "a").replace("Ä", "a").replace("Ö", "o")
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Remove special characters and replace spaces with hyphens
    text = ''.join(c if c.isalnum() else ' ' for c in text)
    text = '-'.join(word for word in text.split() if word)
    
    return text

def eventor_org_result_summary():
    """This is a simple client for integrating with the Eventor orienteering API"""
    
    # Get organisation info
    print("Getting organisation info...")
    org_info = get_organisation()
    print(f"Organisation: {org_info.get('Name', 'Unknown')} with ID {org_info.get('OrganisationId', 'Unknown')}")
    org_id = int(org_info.get("OrganisationId"))
    org_name = org_info.get("Name")
    
    # Get event entries for the last week
    print("\nGetting entries for the last week...")
    today = datetime.now()
    prev_week = today - timedelta(days=7)
    date_range = (
        prev_week.strftime("%Y-%m-%d 00:00:00"),
        today.strftime("%Y-%m-%d 23:59:59")
    )
    
    # Get events where org has entries
    print(f"\nGetting entries for date range: {date_range[0]} to {date_range[1]}")
    events_info = get_events_with_org_entries(
        org_id=org_id,
        from_date=date_range[0],
        to_date=date_range[1]
    )
    print(f"Found {len(events_info)} events with entries from organisation {org_name} (ID: {org_id})")
    
    # fetch results for all events
    results_all_relevant_events = {}
    for event_id, event_data in events_info.items():
        print(f"\nGetting results for event ID: {event_id} - {event_data['eventName']}")
        event_results = get_event_results(event_id, include_split_times=True)
        if not event_results:
            continue
            
        results_data = event_results["results"] if isinstance(event_results, dict) and "results" in event_results else event_results
        splits_data = event_results.get("splits", {}) if isinstance(event_results, dict) else {}
        
        # Create a URL-friendly slug
        event_slug = create_slug(event_data["eventName"])
        
        results_all_relevant_events[event_id] = {
            "eventDescription": {
                "eventName": event_data["eventName"],
                "date": event_data["date"],
                "slug": event_slug
            },
            "classes": results_data,
            "splits": splits_data
        }

    results_all_relevant_events_org = {}
    
    # Add metadata section
    results_all_relevant_events_org["metadata"] = {
        "version": "1.0",
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Eventor API",
        "club": org_name,
        "clubId": str(org_id),
        "dateRange": {
            "from": prev_week.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d")
        },
        "schema": {
            "splitSchema": ["ctrl", "tSec", "pos", "behSec", "totSec"],
            "splits": "Array of arrays following splitSchema format",
            "eventStructure": "Dictionary with event IDs as keys",
            "classes": "Dictionary of class names with participant counts and results"
        }
    }
    
    # Track top performers for the summary
    performer_stats = {}
    
    # Create a more structured events object with descriptive keys
    results_all_relevant_events_org["events"] = {}
    
    for event_id in results_all_relevant_events:
        results_all = results_all_relevant_events[event_id]["classes"]
        splits_all = results_all_relevant_events[event_id].get("splits", {})
        event_data = results_all_relevant_events[event_id]["eventDescription"]
        event_name = event_data["eventName"]
        event_date = event_data["date"]
        
        # Convert date to ISO format YYYY-MM-DD if available
        iso_date = ""
        if event_date and len(event_date) == 8:  # Format: YYYYMMDD
            iso_date = f"{event_date[:4]}-{event_date[4:6]}-{event_date[6:8]}"
        
        # Create a URL-friendly slug for the event key
        event_slug = create_slug(event_name)
        
        total_count = 0
        org_count = 0
        all_results = []  # Flattened array to store all results
        class_counts = {}
        
        for cls_name, class_results in results_all.items():
            total_count += len(class_results)
            class_counts[cls_name] = len(class_results)
            org_class_runners = [runner for runner in class_results if runner["orgId"] == str(org_id)]
            
            if org_class_runners:
                # Add splits data for org members if available
                for runner in org_class_runners:
                    full_name = runner["fullName"]
                    
                    # Add className to each runner result
                    runner["className"] = cls_name
                    runner["classParticipantCount"] = class_counts[cls_name]
                    
                    # Track top performers
                    if "place" in runner and runner["place"] is not None:
                        if full_name not in performer_stats:
                            performer_stats[full_name] = {"name": full_name, "victories": 0, "podiums": 0}
                        
                        # Count victories (1st place)
                        if runner["place"] == 1:
                            performer_stats[full_name]["victories"] += 1
                        
                        # Count podiums (1st, 2nd, 3rd place)
                        if 1 <= runner["place"] <= 3:
                            performer_stats[full_name]["podiums"] += 1
                    
                    # Find this runner's split data in the class splits
                    if cls_name in splits_all:
                        runner_data = [
                            comp for comp in splits_all[cls_name] 
                            if comp["fullName"] == full_name and comp["orgId"] == str(org_id)
                        ]
                        if runner_data and len(runner_data) > 0:
                            # Lets only add compact splits formats for now
                            #runner["splits"] = runner_data[0].get("splits", [])
                            runner["compactSplits"] = runner_data[0].get("compactSplits", [])
                    
                    # Add this runner to the flattened results array
                    all_results.append(runner)
                
                org_count += len(org_class_runners)
        
        if org_count > 0:
            results_all_relevant_events_org["events"][event_slug] = {
                "eventId": event_id,
                "eventDate": iso_date,  # Add ISO formatted date
                "eventDescription": results_all_relevant_events[event_id]["eventDescription"],
                "totalParticipants": total_count,
                "totalStarts": org_count,
                "results": all_results  # Use the flattened results array
            }
        
        event_name = results_all_relevant_events[event_id]["eventDescription"]["eventName"]
        print(f"Number of participants in event {event_id} ({event_name}): {total_count}")
        if org_count > 0:
            print(f"Number of participants from organisation {org_name} (ID: {org_id}): {org_count}")
            print(f"Org results: {len(all_results)} results across all classes")

    # Add summary data to make it AI-friendly
    total_events = len(results_all_relevant_events_org["events"])
    total_club_participants = sum(event_data["totalStarts"] for _, event_data in results_all_relevant_events_org["events"].items())
    event_names = [event_data["eventDescription"]["eventName"] for _, event_data in results_all_relevant_events_org["events"].items()]
    
    # Filter and sort top performers
    top_performers = sorted(
        [stats for _, stats in performer_stats.items() if stats["victories"] > 0 or stats["podiums"] > 0],
        key=lambda x: (x["victories"], x["podiums"]),
        reverse=True
    )[:5]  # Limit to top 5 performers
    
    results_all_relevant_events_org["summary"] = {
        "totalEvents": total_events,
        "totalClubParticipants": total_club_participants,
        "recentCompetitions": event_names,
        "dateRange": f"{prev_week.strftime('%Y%m%d')}-{today.strftime('%Y%m%d')}",
        "topPerformers": top_performers
    }

    # Save results to JSON files
    save_json(results_all_relevant_events, "results_all_relevant_events.json")
    save_json(results_all_relevant_events_org, "results_all_relevant_events_org.json")
    return results_all_relevant_events_org

def eventor_get_recent_events_for_person(daysback =7):
    """Get recent events for a specific person"""
    # Fetch the person ID from the config
    person_id = PERSON_ID
    if not person_id:
        raise ValueError("PERSON_ID is not set in the configuration.")
    
    # Calculate the date range
    toDate = datetime.now()
    fromDate = toDate - timedelta(days=daysback)

    url = f"{API_BASE_URL}/results/person"
    params = {
        "personId": person_id,
        "fromDate": fromDate.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": toDate.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_key = f"raw_{get_cache_key(url, params)}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        print(f"Using cached API data for person {person_id}")
        xml_content = cached_data
    else:
        url_with_params = f"{url}?{urllib.parse.urlencode(params)}"
        response = requests.get(url_with_params, headers=get_headers())
        if response.status_code == 200:
            xml_content = response.content
            save_to_cache(cache_key, xml_content)
        else:
            return None
        
    # Parse the XML content for events and their 
    root = ET.fromstring(xml_content)

    events_info = []
    for event in root.findall('.//Event'):
        event_id = event.findtext('EventId')
        event_name = event.findtext('Name')
        slug_name = create_slug(event_name)
        events_info.append({"eventId": event_id, "eventName": event_name, "slug": slug_name, "personId": person_id})

    # converts event_info to a dictionary with slug as key
    events_info_dict = {event["slug"]: event for event in events_info}

    # Save the events info to a JSON file
    save_json(events_info_dict, "recent_events_for_person.json")
    return events_info_dict

def _add_percent_loss_to_splits(splits, class_splits):
    """
    Adds a 'percent_loss' field to each split in the splits array, comparing to the fastest split for that leg in the class.
    """
    # Build a mapping: controls -> min_time (in seconds) for the class
    min_times = {}
    for comp in class_splits:
        for split in comp.get("splits", []):
            ctrl = split["controls"]
            t = split.get("time")
            if t:
                try:
                    t_sec = int(parse_hms(t).total_seconds())
                    if ctrl not in min_times or t_sec < min_times[ctrl]:
                        min_times[ctrl] = t_sec
                except Exception:
                    pass
    # Add percent_loss to each split
    for split in splits:
        ctrl = split["controls"]
        t = split.get("time")
        percent_loss = None
        if t and ctrl in min_times:
            try:
                t_sec = int(parse_hms(t).total_seconds())
                min_sec = min_times[ctrl]
                if min_sec > 0:
                    percent_loss = round(100 * (t_sec - min_sec) / min_sec, 1)
                else:
                    percent_loss = 0.0
            except Exception:
                percent_loss = None
        split["percent_loss"] = percent_loss
    return splits

def event_get_detailed_result_for_event_and_person(event_id):
    """Get detailed results for a specific event, LLM-friendly and focused on the person of interest"""
    if not event_id:
        raise ValueError("Event ID is not set in the configuration.")

    event_results = get_event_results(event_id, include_split_times=True)
    if not event_results:
        return None

    person_id = PERSON_ID
    if not person_id:
        raise ValueError("PERSON_ID is not set in the configuration.")

    # Find the class and result for the person
    person_result = None
    class_name = None
    class_participant_count = 0
    for cls_name, class_results in event_results["results"].items():
        match = next((res for res in class_results if res["personId"] == str(person_id)), None)
        if match:
            person_result = match
            class_name = cls_name
            class_participant_count = len(class_results)
            break

    if not person_result:
        return None

    # Get splits for the person and all class splits
    splits = None
    compact_splits = None
    class_splits = []
    percent_losses = []
    if class_name in event_results["splits"]:
        class_splits = event_results["splits"][class_name]
        runner_data = [
            comp for comp in class_splits
            if comp["fullName"] == person_result["fullName"] and comp["orgId"] == str(person_result["orgId"])
        ]
        if runner_data:
            splits = runner_data[0].get("splits", [])
            # Add percent_loss to splits
            if splits is not None and class_splits:
                splits = _add_percent_loss_to_splits(splits, class_splits)
                percent_losses = [split.get("percent_loss") for split in splits]
            # Build compact_splits as array of arrays, with percent_loss as 6th value
            compact_splits = []
            for i, cs in enumerate(runner_data[0].get("compactSplits", [])):
                if len(cs) == 5:
                    # Add percent_loss if available, else None
                    percent_loss = percent_losses[i] if i < len(percent_losses) else None
                    compact_splits.append([
                        cs[0], cs[1], cs[2], cs[3], cs[4], percent_loss
                    ])

    # Compose metadata
    metadata = {
        "event_id": event_id,
        "event_name": None,  # Could be filled if event name is available
        "class": class_name,
        "class_participant_count": class_participant_count,
        "person_id": person_result["personId"],
        "person_name": person_result["fullName"],
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": {
            "compact_splits": [
                "controls", "t_sec", "position", "behind_sec", "total_sec", "percent_loss"
            ],
            "splits": [
                "controls", "time", "position", "time_behind", "runningTotal", "percent_loss"
            ]
        }
    }

    # Compose result
    result = {
        k: v for k, v in person_result.items()
        if k not in ("personId", "fullName", "className", "classParticipantCount", "eventId", "compactSplits", "splits")
    }
    result["club"] = person_result.get("club")
    result["org_id"] = person_result.get("orgId")
    result["splits"] = splits if splits else []
    result["compact_splits"] = compact_splits if compact_splits else []

    output = {
        "metadata": metadata,
        "result": result
    }
    save_json(output, f"event_{event_id}_results_for_person_{person_id}.json")
    return output


if __name__ == "__main__":
    eventor_org_result_summary()
    events = eventor_get_recent_events_for_person(daysback=14)  # Call the updated function with the new name and default daysback
    # if events not empty take first event
    if events:
        first_event = list(events.values())[0]
        event_id = first_event["eventId"]
        # Call the detailed result function with the first event ID
    event_get_detailed_result_for_event_and_person(event_id)  # Replace with a valid event ID
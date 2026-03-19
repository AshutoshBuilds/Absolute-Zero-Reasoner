import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__) # Use a logger specific to this module

# Regex for parsing tasks
JSON_CODE_BLOCK_RE = re.compile(r"```json\s*({.+?})\s*```", re.DOTALL)
GENERAL_JSON_RE = re.compile(r"({.+?})", re.DOTALL) # More general JSON block

def find_json_objects(text: str) -> List[str]:
    logger.debug(f"find_json_objects received text (first 500 chars): {text[:500]}")
    json_in_backticks = [match.group(1) for match in JSON_CODE_BLOCK_RE.finditer(text)]
    if json_in_backticks:
        logger.debug(f"Found {len(json_in_backticks)} JSON object(s) in ```json ... ``` blocks.")
        return json_in_backticks
    logger.debug("No ```json ... ``` blocks found, trying fallback brace counting.")

    objects = []
    potential_json_starts = [match.start() for match in re.finditer(r'{', text)]
    logger.debug(f"Fallback: Found {len(potential_json_starts)} potential JSON start braces '{{'.")
    
    for start_index in potential_json_starts:
        brace_count = 0
        for i, char in enumerate(text[start_index:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            
            if brace_count == 0 and i > 0:
                potential_json_segment = text[start_index : start_index + i + 1].strip()
                logger.debug(f"Fallback: Found balanced segment: {potential_json_segment[:200]}...")
                
                if not (potential_json_segment.startswith('{') and potential_json_segment.endswith('}')):
                    logger.debug("Fallback: Segment not starting/ending with braces. Skipping.")
                    break 

                try:
                    json.loads(potential_json_segment) 
                    logger.debug(f"Fallback: Successfully validated segment as JSON: {potential_json_segment[:200]}...")
                    objects.append(potential_json_segment)
                except json.JSONDecodeError as e_direct:
                    logger.debug(f"Fallback: Direct JSON.loads failed for segment: {e_direct}.")
                break 
    if not objects:
        logger.debug("Fallback: No valid JSON objects found after brace counting.")
    else:
        logger.debug(f"Fallback: Found {len(objects)} JSON object(s) via brace counting.")
    return objects

def is_valid_task_structure(task_data: Dict[str, Any], task_idx: int, segment_idx: int) -> bool:
    # task_idx and segment_idx are kept for potential detailed logging if needed, but not used in current validation logic
    required_keys = ["code", "input", "output"]
    if not isinstance(task_data, dict):
        logger.warning(f"Task data is not a dictionary (task_idx: {task_idx}, segment: {segment_idx}): {type(task_data)}")
        return False
    valid = all(key in task_data for key in required_keys)
    if not valid:
        missing_keys = [key for key in required_keys if key not in task_data]
        logger.warning(f"Task data missing required keys (task_idx: {task_idx}, segment: {segment_idx}): {missing_keys}. Data: {task_data}")
    return valid

def parse_generated_tasks(trainer_instance, generated_text: str, task_idx: int) -> list:
    # trainer_instance is passed for consistency but not strictly needed by this parsing function itself.
    # It might be useful if parsing needs access to trainer config or other state in the future.
    parsed_tasks = []
    json_segments = find_json_objects(generated_text)

    if not json_segments:
        logger.warning(f"No JSON objects found in proposer output (task_idx: {task_idx}):\n{generated_text[:500]}...")
        return []

    for i, segment in enumerate(json_segments):
        try:
            task_data = json.loads(segment)
            if isinstance(task_data, list):
                for sub_task_data in task_data:
                    if is_valid_task_structure(sub_task_data, task_idx, segment_idx=i):
                        parsed_tasks.append(sub_task_data)
                    else:
                        logger.warning(f"Skipping malformed task object (task_idx: {task_idx}, segment: {i}, sub_task from list): {sub_task_data}")
            elif isinstance(task_data, dict):
                if is_valid_task_structure(task_data, task_idx, segment_idx=i):
                    parsed_tasks.append(task_data)
                else:
                    logger.warning(f"Skipping malformed task object (task_idx: {task_idx}, segment: {i}): {task_data}")
            else:
                logger.warning(f"Parsed JSON segment is not a dict or list (task_idx: {task_idx}, segment: {i}): {type(task_data)}. Segment: {segment[:200]}...")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for segment (task_idx: {task_idx}, segment_idx: {i}): '{segment[:200]}...'. Error: {e.msg}")
            logger.warning(f"  Error details: At line {e.lineno}, column {e.colno} (char {e.pos}).")
            if len(e.doc) < 1000:
                logger.warning(f"  Full problematic segment content: {e.doc}")
            else:
                context_start = max(0, e.pos - 150)
                context_end = min(len(e.doc), e.pos + 150)
                logger.warning(f"  Snippet of problematic segment (char {context_start}-{context_end}): '{e.doc[context_start:context_end]}'")

    if not parsed_tasks and json_segments:
        logger.warning(f"Found JSON string(s) but failed to parse into valid task structures (task_idx: {task_idx}). First JSON string: {json_segments[0][:200]}...")
    
    return parsed_tasks 
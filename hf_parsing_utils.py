import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__) # Use a logger specific to this module

# Strict ```json ... ``` fences (proposer must use this for structured tasks).
JSON_JSON_FENCE_RE = re.compile(r"```\s*json\s*([\s\S]*?)```", re.IGNORECASE)
# Any fenced block: language tag (group 1) + body (group 2). Used only after strict json fences fail.
GENERIC_FENCE_RE = re.compile(r"```\s*([a-zA-Z0-9_+-]*)\s*([\s\S]*?)```", re.IGNORECASE)
# Languages that are almost never JSON task payloads — skip to avoid ```python { ... }``` false tries.
_FENCE_LANG_SKIP = frozenset(
    {
        "python",
        "py",
        "bash",
        "sh",
        "shell",
        "zsh",
        "pwsh",
        "powershell",
        "text",
        "plaintext",
        "markdown",
        "md",
        "diff",
        "yaml",
        "yml",
        "toml",
        "rust",
        "c",
        "cpp",
        "java",
        "go",
        "ts",
        "tsx",
        "jsx",
        "html",
        "css",
        "sql",
    }
)

# Entire snippet is a single ```lang ... ``` block (solver / executor hygiene).
_OUTER_CODE_FENCE_RE = re.compile(
    r"^\s*```[ \t]*([a-zA-Z0-9_+-]*)[ \t]*(?:\r?\n)([\s\S]*?)\r?\n[ \t]*```[ \t]*\s*$",
    re.IGNORECASE,
)


def strip_leading_trailing_code_fences(text: str) -> str:
    """
    Strip one layer of markdown code fences so ``exec``/AST checks see valid Python.

    Handles a full ```python / ```json / plain ``` wrapper, or a leading opening fence
    line plus a trailing closing fence, as models often emit around solver output.
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return s
    m = _OUTER_CODE_FENCE_RE.match(s)
    if m:
        return m.group(2).strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl == -1:
            return s[3:].strip()
        rest = s[nl + 1 :]
        end = rest.rfind("```")
        if end != -1:
            return rest[:end].strip()
        return rest.strip()
    return s


def _strip_json_like_comments(text: str) -> str:
    """Strip line ('//') and block ('/* */') comments while preserving JSON strings."""
    if not text:
        return text

    output = []
    i = 0
    in_string = False
    escaped = False
    quote_char = ""

    while i < len(text):
        char = text[i]

        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            i += 1
            continue

        if char in ("'", '"'):
            in_string = True
            quote_char = char
            output.append(char)
            i += 1
            continue

        if char == "/" and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char == "/":
                i += 2
                while i < len(text) and text[i] not in "\r\n":
                    i += 1
                continue
            if next_char == "*":
                end_idx = text.find("*/", i + 2)
                if end_idx == -1:
                    return "".join(output)
                i = end_idx + 2
                continue

        output.append(char)
        i += 1

    return "".join(output)


def _find_balanced_json_segments(text: str) -> List[str]:
    """Find balanced JSON object/array segments using brace counting, ignoring comments and strings."""
    if not text:
        return []

    candidates = []
    candidate_starts = [idx for idx, char in enumerate(text) if char in ("{", "[")]
    if not candidate_starts:
        return candidates

    for start_index in candidate_starts:
        stack: List[str] = []
        in_string = False
        escaped = False
        quote_char = ""

        for i in range(start_index, len(text)):
            char = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote_char:
                    in_string = False
                continue

            if char in ("'", '"'):
                in_string = True
                quote_char = char
                continue

            if char in "{[":
                stack.append(char)
                continue

            if not stack:
                continue

            if char == "}" and stack[-1] == "{":
                stack.pop()
            elif char == "]" and stack[-1] == "[":
                stack.pop()
            elif char in "}]":
                # Broken JSON segment; stop this start.
                stack = []
                break

            if not stack:
                segment = text[start_index : i + 1].strip()
                if segment:
                    candidates.append(segment)
                break

    return candidates


def _first_loadable_json_string(raw: str) -> Optional[str]:
    """
    Return the first substring of ``raw`` that ``json.loads`` accepts: the full stripped
    string, or the first balanced `{`/`[` segment inside it.
    """
    if not raw or not str(raw).strip():
        return None
    stripped = str(raw).strip()
    candidates: List[str] = [stripped]
    candidates.extend(_find_balanced_json_segments(stripped))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def _normalize_task_fields(task_data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(task_data, dict):
        logger.warning(
            "Skipping non-object task entry (expected dict, got %s): %r",
            type(task_data).__name__,
            task_data,
        )
        return None
    normalized_task_data = dict(task_data)
    if "code" not in normalized_task_data and "description" in normalized_task_data:
        normalized_task_data["code"] = normalized_task_data["description"]
    if "output" not in normalized_task_data and "equivalent" in normalized_task_data:
        normalized_task_data["output"] = normalized_task_data["equivalent"]
    return normalized_task_data

def find_json_objects(text: str) -> List[str]:
    logger.debug(f"find_json_objects received text (first 500 chars): {text[:500]}")
    text = _strip_json_like_comments(text)

    validated: List[str] = []
    for match in JSON_JSON_FENCE_RE.finditer(text):
        body = match.group(1).strip()
        body = _strip_json_like_comments(body)
        payload = _first_loadable_json_string(body)
        if payload is not None:
            validated.append(payload)

    if validated:
        logger.debug("Found %s JSON payload(s) in ```json ... ``` fences.", len(validated))
        return validated

    logger.debug("No valid JSON in ```json``` fences; trying generic fenced blocks (non-code languages).")
    for match in GENERIC_FENCE_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        if lang in _FENCE_LANG_SKIP:
            continue
        if lang not in ("", "json", "javascript", "js"):
            continue
        body = match.group(2).strip()
        body = _strip_json_like_comments(body)
        payload = _first_loadable_json_string(body)
        if payload is not None:
            validated.append(payload)

    if validated:
        logger.debug("Found %s JSON payload(s) in generic ``` fences.", len(validated))
        return validated

    logger.debug("No fenced JSON blocks found, trying balanced JSON recovery.")

    objects = []
    potential_json_segments = _find_balanced_json_segments(text)
    logger.debug(f"Fallback: Found {len(potential_json_segments)} balanced JSON segment candidate(s).")

    for start_index, segment in enumerate(potential_json_segments):
        logger.debug(f"Fallback: Balanced candidate[{start_index}] = {segment[:200]}...")
        try:
            json.loads(segment)
            logger.debug(f"Fallback: Successfully validated segment as JSON: {segment[:200]}...")
            objects.append(segment)
        except json.JSONDecodeError as e_direct:
            logger.debug(f"Fallback: JSON.loads failed for candidate: {e_direct}.")
    if not objects:
        logger.debug("Fallback: No valid JSON objects found after brace counting.")
    else:
        logger.debug(f"Fallback: Found {len(objects)} JSON object(s) via brace counting.")
    return objects

def is_valid_task_structure(task_data: Dict[str, Any], task_idx: int, segment_idx: int) -> bool:
    # task_idx and segment_idx are kept for potential detailed logging if needed, but not used in current validation logic
    if not isinstance(task_data, dict):
        logger.warning(f"Task data is not a dictionary (task_idx: {task_idx}, segment: {segment_idx}): {type(task_data)}")
        return False

    code_field_present = "code" in task_data or "description" in task_data
    output_field_present = "output" in task_data or "equivalent" in task_data
    missing_keys = []
    if "input" not in task_data:
        missing_keys.append("input")
    if not code_field_present:
        missing_keys.append("code|description")
    if not output_field_present:
        missing_keys.append("output|equivalent")

    valid = len(missing_keys) == 0
    if not valid:
        logger.warning(f"Task data missing required keys (task_idx: {task_idx}, segment: {segment_idx}): {missing_keys}. Data: {task_data}")
    return valid

def parse_generated_tasks(trainer_instance, generated_text: str, task_idx: int) -> list:
    # trainer_instance is passed for consistency but not strictly needed by this parsing function itself.
    # It might be useful if parsing needs access to trainer config or other state in the future.
    parsed_tasks = []

    if generated_text is None:
        logger.warning(f"Null proposer output (task_idx: {task_idx}).")
        return []

    stripped_output = generated_text.strip()
    try:
        task_data = json.loads(stripped_output)
        strict_tasks = []
        if isinstance(task_data, list):
            strict_tasks = task_data
        elif isinstance(task_data, dict):
            strict_tasks = [task_data]

        if strict_tasks:
            for task in strict_tasks:
                normalized_task = _normalize_task_fields(task)
                if normalized_task is None:
                    continue
                if is_valid_task_structure(normalized_task, task_idx, segment_idx=0):
                    parsed_tasks.append(normalized_task)
                else:
                    logger.warning(
                        f"Skipping malformed strict task object (task_idx: {task_idx}, segment: 0): {task}"
                    )
            if parsed_tasks:
                return parsed_tasks
            logger.debug(
                f"Strict JSON parse succeeded for task_idx={task_idx} but task structure was invalid. Falling back to recovery."
            )
    except json.JSONDecodeError as e:
        logger.debug(f"Strict JSON parse failed for task_idx: {task_idx}: {e.msg}")

    json_segments = find_json_objects(stripped_output)

    if not json_segments:
        logger.warning(f"No JSON objects found in proposer output (task_idx: {task_idx}):\n{generated_text[:500]}...")
        return []

    for i, segment in enumerate(json_segments):
        try:
            task_data = json.loads(segment)
            if isinstance(task_data, list):
                for sub_task_data in task_data:
                    normalized_task = _normalize_task_fields(sub_task_data)
                    if normalized_task is None:
                        continue
                    if is_valid_task_structure(normalized_task, task_idx, segment_idx=i):
                        parsed_tasks.append(normalized_task)
                    else:
                        logger.warning(
                            f"Skipping malformed task object (task_idx: {task_idx}, segment: {i}, sub_task from list): {sub_task_data}"
                        )
            elif isinstance(task_data, dict):
                normalized_task = _normalize_task_fields(task_data)
                if is_valid_task_structure(normalized_task, task_idx, segment_idx=i):
                    parsed_tasks.append(normalized_task)
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
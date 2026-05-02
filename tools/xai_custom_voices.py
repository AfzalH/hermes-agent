"""xAI Custom Voices API client.

Manage custom voice clones for use with xAI TTS and Voice Agent APIs.

Custom voices are created from a short reference audio clip (30–120 seconds)
and can be used anywhere a built-in xAI voice works — REST TTS, streaming
TTS WebSocket, and the real-time Voice Agent.

Endpoints:
  - POST   /v1/custom-voices          — Create (Enterprise-only via API)
  - GET    /v1/custom-voices           — List all team voices
  - GET    /v1/custom-voices/{id}      — Get metadata for a voice
  - PATCH  /v1/custom-voices/{id}      — Update metadata
  - DELETE /v1/custom-voices/{id}      — Delete a voice
  - GET    /v1/custom-voices/{id}/audio — Download reference audio

Custom voices are scoped to the team and never visible to other users.
They are returned only by GET /v1/custom-voices — not in the built-in
voice list (GET /v1/tts/voices).

Usage::

    from tools.xai_custom_voices import (
        list_custom_voices,
        get_custom_voice,
        create_custom_voice,
        update_custom_voice,
        delete_custom_voice,
        download_custom_voice_audio,
        list_builtin_voices,
    )
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.x.ai/v1"
CUSTOM_VOICES_PATH = "/custom-voices"

# Valid field values per xAI docs
VALID_GENDERS = frozenset({"male", "female", "neutral"})
VALID_AGES = frozenset({"young", "middle-aged", "old"})
VALID_USE_CASES = frozenset({
    "conversational", "narration", "characters",
    "educational", "advertisement", "social_media", "entertainment",
})
VALID_TONES = frozenset({
    "warm", "casual", "professional", "friendly",
    "authoritative", "expressive", "calm",
})


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read env values through the live config module."""
    try:
        from hermes_cli.config import get_env_value as _get_env
    except ImportError:
        return os.getenv(name, default)
    value = _get_env(name)
    return default if value is None else value


def _get_api_key() -> str:
    """Return the xAI API key or raise ValueError."""
    api_key = (_get_env_value("XAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("XAI_API_KEY not set. Get one at https://console.x.ai/")
    return api_key


def _get_base_url() -> str:
    """Return the xAI base URL from config or default."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        xai_config = config.get("tts", {}).get("xai", {})
        url = xai_config.get("base_url")
        if url:
            return str(url).strip().rstrip("/")
    except Exception:
        pass
    env_url = _get_env_value("XAI_BASE_URL")
    if env_url:
        return env_url.strip().rstrip("/")
    return DEFAULT_BASE_URL


def _headers(api_key: str) -> Dict[str, str]:
    """Common request headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": hermes_xai_user_agent(),
    }


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------


def list_custom_voices(
    limit: int = 100,
    pagination_token: Optional[str] = None,
) -> Dict[str, Any]:
    """List all custom voices for the team.

    Args:
        limit: Page size (1–1000, default 100).
        pagination_token: Token from a previous response for pagination.

    Returns:
        API response dict with voice list and optional pagination_token.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    params: Dict[str, Any] = {"limit": limit}
    if pagination_token:
        params["pagination_token"] = pagination_token

    response = requests.get(
        f"{base_url}{CUSTOM_VOICES_PATH}",
        headers=_headers(api_key),
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_custom_voice(voice_id: str) -> Dict[str, Any]:
    """Get metadata for a single custom voice.

    Args:
        voice_id: The 8-character voice identifier.

    Returns:
        Voice metadata dict.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    response = requests.get(
        f"{base_url}{CUSTOM_VOICES_PATH}/{voice_id}",
        headers=_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def create_custom_voice(
    file_path: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    gender: Optional[str] = None,
    accent: Optional[str] = None,
    age: Optional[str] = None,
    language: Optional[str] = None,
    use_case: Optional[str] = None,
    tone: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a custom voice from a reference audio file.

    Note: API creation is Enterprise-only. Console creation is available
    to all users at https://console.x.ai/team/default/voice/voice-library

    Args:
        file_path: Path to reference audio file (max 120 seconds).
        name: Display name for the voice.
        description: Free-text description.
        gender: One of 'male', 'female', 'neutral'.
        accent: Free text (e.g. 'British', 'American').
        age: One of 'young', 'middle-aged', 'old'.
        language: ISO 639 or BCP-47 code (e.g. 'en', 'en-US').
        use_case: One of the valid use cases.
        tone: One of the valid tones.

    Returns:
        Created voice metadata dict including voice_id.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Reference audio file not found: {file_path}")

    # Build form data with only non-None fields
    data: Dict[str, str] = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if gender is not None:
        if gender not in VALID_GENDERS:
            raise ValueError(f"Invalid gender '{gender}'. Must be one of: {sorted(VALID_GENDERS)}")
        data["gender"] = gender
    if accent is not None:
        data["accent"] = accent
    if age is not None:
        if age not in VALID_AGES:
            raise ValueError(f"Invalid age '{age}'. Must be one of: {sorted(VALID_AGES)}")
        data["age"] = age
    if language is not None:
        data["language"] = language
    if use_case is not None:
        if use_case not in VALID_USE_CASES:
            raise ValueError(f"Invalid use_case '{use_case}'. Must be one of: {sorted(VALID_USE_CASES)}")
        data["use_case"] = use_case
    if tone is not None:
        if tone not in VALID_TONES:
            raise ValueError(f"Invalid tone '{tone}'. Must be one of: {sorted(VALID_TONES)}")
        data["tone"] = tone

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{base_url}{CUSTOM_VOICES_PATH}",
            headers=_headers(api_key),
            files={"file": (audio_path.name, f)},
            data=data,
            timeout=120,
        )

    response.raise_for_status()
    result = response.json()
    logger.info("Created custom voice '%s' (id=%s)", result.get("name", ""), result.get("voice_id", ""))
    return result


def update_custom_voice(
    voice_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    gender: Optional[str] = None,
    accent: Optional[str] = None,
    age: Optional[str] = None,
    language: Optional[str] = None,
    use_case: Optional[str] = None,
    tone: Optional[str] = None,
) -> Dict[str, Any]:
    """Update metadata for a custom voice.

    Fields set to a non-empty string update the value.
    Fields set to None are left unchanged (omitted from the request).
    To clear a field, the xAI API expects the value set to null — pass
    the sentinel string '__clear__' to indicate clearing.

    Note: Cannot change the underlying audio. To re-record, delete and
    create a new voice.

    Args:
        voice_id: The 8-character voice identifier.
        name: New display name (or '__clear__' to clear).
        description: New description (or '__clear__' to clear).
        gender: New gender value (or '__clear__' to clear).
        accent: New accent (or '__clear__' to clear).
        age: New age value (or '__clear__' to clear).
        language: New language code (or '__clear__' to clear).
        use_case: New use case (or '__clear__' to clear).
        tone: New tone (or '__clear__' to clear).

    Returns:
        Updated voice metadata dict.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    CLEAR_SENTINEL = "__clear__"
    payload: Dict[str, Any] = {}

    for field_name, value, valid_set in [
        ("name", name, None),
        ("description", description, None),
        ("gender", gender, VALID_GENDERS),
        ("accent", accent, None),
        ("age", age, VALID_AGES),
        ("language", language, None),
        ("use_case", use_case, VALID_USE_CASES),
        ("tone", tone, VALID_TONES),
    ]:
        if value is None:
            continue
        if value == CLEAR_SENTINEL:
            payload[field_name] = None
        else:
            if valid_set and value not in valid_set:
                raise ValueError(f"Invalid {field_name} '{value}'. Must be one of: {sorted(valid_set)}")
            payload[field_name] = value

    if not payload:
        raise ValueError("No fields to update. Provide at least one field to change.")

    response = requests.patch(
        f"{base_url}{CUSTOM_VOICES_PATH}/{voice_id}",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    logger.info("Updated custom voice '%s' (id=%s)", result.get("name", ""), voice_id)
    return result


def delete_custom_voice(voice_id: str) -> bool:
    """Delete a custom voice.

    Args:
        voice_id: The 8-character voice identifier.

    Returns:
        True if successfully deleted.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    response = requests.delete(
        f"{base_url}{CUSTOM_VOICES_PATH}/{voice_id}",
        headers=_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    logger.info("Deleted custom voice (id=%s)", voice_id)
    return True


def download_custom_voice_audio(voice_id: str, output_path: str) -> str:
    """Download the reference audio for a custom voice.

    Args:
        voice_id: The 8-character voice identifier.
        output_path: Where to save the downloaded audio file.

    Returns:
        Path to the saved audio file.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    response = requests.get(
        f"{base_url}{CUSTOM_VOICES_PATH}/{voice_id}/audio",
        headers=_headers(api_key),
        timeout=60,
        stream=True,
    )
    response.raise_for_status()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info("Downloaded reference audio for voice %s to %s", voice_id, output_path)
    return str(output)


def list_builtin_voices() -> Dict[str, Any]:
    """List all built-in xAI TTS voices.

    Returns:
        API response dict with voice list. Each voice has voice_id, name,
        language, gender, accent, age, use_case, and description fields.
    """
    import requests

    api_key = _get_api_key()
    base_url = _get_base_url()

    response = requests.get(
        f"{base_url}/tts/voices",
        headers=_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

"""Tests for tools/xai_custom_voices.py — xAI Custom Voices API client."""

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no XAI env vars leak between tests."""
    for key in ("XAI_API_KEY", "XAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def set_api_key(monkeypatch):
    """Set a fake XAI API key."""
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key-12345")


@pytest.fixture
def sample_audio(tmp_path):
    """Create a minimal sample audio file."""
    audio_file = tmp_path / "reference.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
    return str(audio_file)


@pytest.fixture
def mock_load_config():
    """Patch hermes_cli.config.load_config to return empty config."""
    with patch("tools.xai_custom_voices._get_env_value") as mock_env:
        mock_env.side_effect = lambda name, default=None: {
            "XAI_API_KEY": "xai-test-key-12345",
            "XAI_BASE_URL": None,
        }.get(name, default)
        yield mock_env


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_missing_key_raises(self, monkeypatch):
        from tools.xai_custom_voices import _get_api_key

        with patch("tools.xai_custom_voices._get_env_value", return_value=""):
            with pytest.raises(ValueError, match="XAI_API_KEY not set"):
                _get_api_key()

    def test_key_from_env(self, set_api_key):
        from tools.xai_custom_voices import _get_api_key

        key = _get_api_key()
        assert key == "xai-test-key-12345"


# ---------------------------------------------------------------------------
# list_custom_voices
# ---------------------------------------------------------------------------


class TestListCustomVoices:
    def test_no_api_key_raises(self):
        from tools.xai_custom_voices import list_custom_voices

        with patch("tools.xai_custom_voices._get_api_key", side_effect=ValueError("XAI_API_KEY not set")):
            with pytest.raises(ValueError, match="XAI_API_KEY"):
                list_custom_voices()

    def test_successful_list(self, set_api_key):
        from tools.xai_custom_voices import list_custom_voices

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "voices": [
                {"voice_id": "abc12345", "name": "My Voice", "gender": "male"},
                {"voice_id": "def67890", "name": "Other Voice", "gender": "female"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = list_custom_voices(limit=50)

        assert len(result["voices"]) == 2
        assert result["voices"][0]["voice_id"] == "abc12345"
        # Verify correct URL and params
        call_args = mock_get.call_args
        assert "/custom-voices" in call_args[0][0] or "/custom-voices" in str(call_args)
        assert call_args[1]["params"]["limit"] == 50

    def test_pagination_token_passed(self, set_api_key):
        from tools.xai_custom_voices import list_custom_voices

        mock_response = MagicMock()
        mock_response.json.return_value = {"voices": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            list_custom_voices(pagination_token="next_page_token")

        call_args = mock_get.call_args
        assert call_args[1]["params"]["pagination_token"] == "next_page_token"


# ---------------------------------------------------------------------------
# get_custom_voice
# ---------------------------------------------------------------------------


class TestGetCustomVoice:
    def test_successful_get(self, set_api_key):
        from tools.xai_custom_voices import get_custom_voice

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "voice_id": "nlbqfwie",
            "name": "Friendly Narrator",
            "gender": "female",
            "tone": "warm",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = get_custom_voice("nlbqfwie")

        assert result["voice_id"] == "nlbqfwie"
        assert result["name"] == "Friendly Narrator"
        assert "nlbqfwie" in mock_get.call_args[0][0]

    def test_not_found_raises(self, set_api_key):
        from tools.xai_custom_voices import get_custom_voice
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                get_custom_voice("nonexist")


# ---------------------------------------------------------------------------
# create_custom_voice
# ---------------------------------------------------------------------------


class TestCreateCustomVoice:
    def test_file_not_found_raises(self, set_api_key):
        from tools.xai_custom_voices import create_custom_voice

        with pytest.raises(FileNotFoundError, match="not found"):
            create_custom_voice("/nonexistent/audio.wav")

    def test_successful_create(self, set_api_key, sample_audio):
        from tools.xai_custom_voices import create_custom_voice

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "voice_id": "newvoice",
            "name": "Test Voice",
            "gender": "male",
            "language": "en",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = create_custom_voice(
                sample_audio,
                name="Test Voice",
                gender="male",
                language="en",
                tone="warm",
            )

        assert result["voice_id"] == "newvoice"
        call_kwargs = mock_post.call_args[1]
        assert "files" in call_kwargs
        assert call_kwargs["data"]["name"] == "Test Voice"
        assert call_kwargs["data"]["gender"] == "male"
        assert call_kwargs["data"]["language"] == "en"
        assert call_kwargs["data"]["tone"] == "warm"

    def test_invalid_gender_raises(self, set_api_key, sample_audio):
        from tools.xai_custom_voices import create_custom_voice

        with pytest.raises(ValueError, match="Invalid gender"):
            create_custom_voice(sample_audio, gender="robot")

    def test_invalid_age_raises(self, set_api_key, sample_audio):
        from tools.xai_custom_voices import create_custom_voice

        with pytest.raises(ValueError, match="Invalid age"):
            create_custom_voice(sample_audio, age="ancient")

    def test_invalid_use_case_raises(self, set_api_key, sample_audio):
        from tools.xai_custom_voices import create_custom_voice

        with pytest.raises(ValueError, match="Invalid use_case"):
            create_custom_voice(sample_audio, use_case="singing")

    def test_invalid_tone_raises(self, set_api_key, sample_audio):
        from tools.xai_custom_voices import create_custom_voice

        with pytest.raises(ValueError, match="Invalid tone"):
            create_custom_voice(sample_audio, tone="angry")

    def test_optional_fields_omitted_when_none(self, set_api_key, sample_audio):
        from tools.xai_custom_voices import create_custom_voice

        mock_response = MagicMock()
        mock_response.json.return_value = {"voice_id": "abcd1234"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            create_custom_voice(sample_audio)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["data"] == {}


# ---------------------------------------------------------------------------
# update_custom_voice
# ---------------------------------------------------------------------------


class TestUpdateCustomVoice:
    def test_successful_update(self, set_api_key):
        from tools.xai_custom_voices import update_custom_voice

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "voice_id": "nlbqfwie",
            "name": "Updated Name",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.patch", return_value=mock_response) as mock_patch:
            result = update_custom_voice("nlbqfwie", name="Updated Name")

        assert result["name"] == "Updated Name"
        call_kwargs = mock_patch.call_args[1]
        assert call_kwargs["json"] == {"name": "Updated Name"}

    def test_clear_field_sends_null(self, set_api_key):
        from tools.xai_custom_voices import update_custom_voice

        mock_response = MagicMock()
        mock_response.json.return_value = {"voice_id": "nlbqfwie"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.patch", return_value=mock_response) as mock_patch:
            update_custom_voice("nlbqfwie", description="__clear__")

        call_kwargs = mock_patch.call_args[1]
        assert call_kwargs["json"] == {"description": None}

    def test_no_fields_raises(self, set_api_key):
        from tools.xai_custom_voices import update_custom_voice

        with pytest.raises(ValueError, match="No fields to update"):
            update_custom_voice("nlbqfwie")

    def test_invalid_gender_raises(self, set_api_key):
        from tools.xai_custom_voices import update_custom_voice

        with pytest.raises(ValueError, match="Invalid gender"):
            update_custom_voice("nlbqfwie", gender="robot")

    def test_invalid_tone_raises(self, set_api_key):
        from tools.xai_custom_voices import update_custom_voice

        with pytest.raises(ValueError, match="Invalid tone"):
            update_custom_voice("nlbqfwie", tone="angry")


# ---------------------------------------------------------------------------
# delete_custom_voice
# ---------------------------------------------------------------------------


class TestDeleteCustomVoice:
    def test_successful_delete(self, set_api_key):
        from tools.xai_custom_voices import delete_custom_voice

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status = MagicMock()

        with patch("requests.delete", return_value=mock_response):
            result = delete_custom_voice("nlbqfwie")

        assert result is True

    def test_not_found_raises(self, set_api_key):
        from tools.xai_custom_voices import delete_custom_voice
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )

        with patch("requests.delete", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                delete_custom_voice("nonexist")


# ---------------------------------------------------------------------------
# download_custom_voice_audio
# ---------------------------------------------------------------------------


class TestDownloadCustomVoiceAudio:
    def test_successful_download(self, set_api_key, tmp_path):
        from tools.xai_custom_voices import download_custom_voice_audio

        audio_content = b"RIFF" + b"\xff" * 200

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [audio_content]
        mock_response.raise_for_status = MagicMock()

        output_path = str(tmp_path / "downloaded.wav")

        with patch("requests.get", return_value=mock_response):
            result = download_custom_voice_audio("nlbqfwie", output_path)

        assert result == output_path
        assert Path(output_path).exists()
        assert Path(output_path).read_bytes() == audio_content

    def test_creates_parent_dirs(self, set_api_key, tmp_path):
        from tools.xai_custom_voices import download_custom_voice_audio

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"audio"]
        mock_response.raise_for_status = MagicMock()

        output_path = str(tmp_path / "nested" / "dir" / "audio.wav")

        with patch("requests.get", return_value=mock_response):
            download_custom_voice_audio("nlbqfwie", output_path)

        assert Path(output_path).exists()


# ---------------------------------------------------------------------------
# _get_base_url
# ---------------------------------------------------------------------------


class TestGetBaseUrl:
    def test_default_url(self):
        from tools.xai_custom_voices import _get_base_url, DEFAULT_BASE_URL

        with patch("tools.xai_custom_voices._get_env_value", return_value=None):
            try:
                with patch("hermes_cli.config.load_config", return_value={}):
                    url = _get_base_url()
            except ImportError:
                url = _get_base_url()

        assert url == DEFAULT_BASE_URL

    def test_env_override(self, monkeypatch):
        from tools.xai_custom_voices import _get_base_url

        monkeypatch.setenv("XAI_BASE_URL", "https://custom.x.ai/v1")
        url = _get_base_url()
        assert url == "https://custom.x.ai/v1"

    def test_trailing_slash_stripped(self, monkeypatch):
        from tools.xai_custom_voices import _get_base_url

        monkeypatch.setenv("XAI_BASE_URL", "https://custom.x.ai/v1/")
        url = _get_base_url()
        assert not url.endswith("/")


# ---------------------------------------------------------------------------
# _headers
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_includes_auth_and_user_agent(self):
        from tools.xai_custom_voices import _headers

        with patch("tools.xai_custom_voices.hermes_xai_user_agent", return_value="Hermes-Agent/1.0"):
            h = _headers("test-key")

        assert h["Authorization"] == "Bearer test-key"
        assert h["User-Agent"] == "Hermes-Agent/1.0"


# ---------------------------------------------------------------------------
# list_builtin_voices
# ---------------------------------------------------------------------------


class TestListBuiltinVoices:
    def test_successful_list(self, set_api_key):
        from tools.xai_custom_voices import list_builtin_voices

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "voices": [
                {"voice_id": "eve", "name": "Eve", "language": "multilingual"},
                {"voice_id": "leo", "name": "Leo", "language": "multilingual"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = list_builtin_voices()

        assert len(result["voices"]) == 2
        assert result["voices"][0]["voice_id"] == "eve"
        # Verify it hits /tts/voices not /custom-voices
        call_url = mock_get.call_args[0][0]
        assert "/tts/voices" in call_url
        assert "/custom-voices" not in call_url

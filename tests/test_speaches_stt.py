from types import SimpleNamespace

import pytest

from pipecat.services.speaches.stt import SpeachesSTTService, SpeachesSTTSettings


@pytest.mark.asyncio
async def test_speaches_stt_uses_openai_compatible_transcription_request():
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="Merhaba")

    service = SpeachesSTTService(
        api_key="none",
        base_url="http://localhost:9100/v1",
        settings=SpeachesSTTSettings(
            model="Systran/faster-whisper-small",
            language="tr",
        ),
    )
    service._client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=create,
            )
        )
    )

    result = await service._transcribe(b"wav-bytes")

    assert result.text == "Merhaba"
    assert captured["file"] == ("audio.wav", b"wav-bytes", "audio/wav")
    assert captured["model"] == "Systran/faster-whisper-small"
    assert captured["language"] == "tr"

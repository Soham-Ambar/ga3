import base64
import binascii
import json
import os
import tempfile
from typing import Any

from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel


router = APIRouter()

REQUIRED_KEYS = [
    "rows",
    "columns",
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "median",
    "mode",
    "range",
    "allowed_values",
    "value_range",
    "correlation",
]


class AudioStatisticsRequest(BaseModel):
    audio_id: str
    audio_base64: str


class AudioStatisticsResponse(BaseModel):
    rows: int
    columns: list[Any]
    mean: dict[str, Any]
    std: dict[str, Any]
    variance: dict[str, Any]
    min: dict[str, Any]
    max: dict[str, Any]
    median: dict[str, Any]
    mode: dict[str, Any]
    range: dict[str, Any]
    allowed_values: dict[str, Any]
    value_range: dict[str, Any]
    correlation: list[Any]


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured",
        )

    return Groq(
        api_key=api_key,
        timeout=120,
        max_retries=2,
    )


def decode_audio(audio_base64: str) -> bytes:
    encoded = audio_base64.strip()

    # Support data URLs:
    # data:audio/wav;base64,UklGR...
    if encoded.lower().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    encoded = "".join(encoded.split())

    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="audio_base64 is not valid base64",
        ) from error

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Decoded audio is empty",
        )

    return audio_bytes


def detect_audio_extension(audio_bytes: bytes) -> str:
    """
    Detect the actual audio format using file signatures.
    """

    # WAV: RIFF....WAVE
    if (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    ):
        return ".wav"

    # MP3 with ID3 metadata
    if audio_bytes[:3] == b"ID3":
        return ".mp3"

    # MP3 frame header
    if len(audio_bytes) >= 2:
        first = audio_bytes[0]
        second = audio_bytes[1]

        if first == 0xFF and (second & 0xE0) == 0xE0:
            return ".mp3"

    # FLAC
    if audio_bytes[:4] == b"fLaC":
        return ".flac"

    # OGG
    if audio_bytes[:4] == b"OggS":
        return ".ogg"

    # WebM / Matroska
    if audio_bytes[:4] == bytes.fromhex("1A45DFA3"):
        return ".webm"

    # MP4, M4A and related containers
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".m4a"

    # Use WAV as a safe filename fallback.
    return ".wav"


def transcribe_audio(audio_bytes: bytes) -> str:
    client = get_groq_client()
    extension = detect_audio_extension(audio_bytes)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        with open(temporary_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(
                    f"korean_audio{extension}",
                    audio_file,
                ),
                model="whisper-large-v3-turbo",
                language="ko",
                response_format="json",
                temperature=0,
            )

        transcript = getattr(transcription, "text", None)

        if not transcript and isinstance(transcription, dict):
            transcript = transcription.get("text")

        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="Audio transcription was empty",
            )

        return transcript.strip()

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"Q6 transcription error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Audio transcription failed: {str(error)}",
        ) from error

    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def clean_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(
            f"Q6 invalid model JSON: {cleaned}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Statistics model returned invalid JSON",
        ) from error

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Statistics model did not return an object",
        )

    return result


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    missing_keys = set(REQUIRED_KEYS) - set(result.keys())

    if missing_keys:
        raise HTTPException(
            status_code=500,
            detail=f"Missing result keys: {sorted(missing_keys)}",
        )

    # Rebuild the dictionary so no extra keys are returned.
    normalized = {
        "rows": result["rows"],
        "columns": result["columns"],
        "mean": result["mean"],
        "std": result["std"],
        "variance": result["variance"],
        "min": result["min"],
        "max": result["max"],
        "median": result["median"],
        "mode": result["mode"],
        "range": result["range"],
        "allowed_values": result["allowed_values"],
        "value_range": result["value_range"],
        "correlation": result["correlation"],
    }

    try:
        validated = AudioStatisticsResponse.model_validate(normalized)
    except Exception as error:
        print(
            f"Q6 validation error: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Incorrect result data types: {str(error)}",
        ) from error

    return validated.model_dump()


def analyze_transcript(
    audio_id: str,
    transcript: str,
) -> dict[str, Any]:
    client = get_groq_client()

    system_prompt = """
You are a precise Korean-language dataset analysis system.

The user provides a transcript of Korean audio describing a dataset. Extract
all records, columns, values, constraints, categories and requested
statistics from the transcript.

Return exactly one JSON object with exactly these top-level keys:

{
  "rows": 0,
  "columns": [],
  "mean": {},
  "std": {},
  "variance": {},
  "min": {},
  "max": {},
  "median": {},
  "mode": {},
  "range": {},
  "allowed_values": {},
  "value_range": {},
  "correlation": []
}

Rules:

1. rows is the number of data rows, excluding the header.
2. columns preserves the requested column order.
3. Calculate numeric statistics accurately.
4. Use pandas-compatible sample standard deviation and sample variance:
   std uses ddof=1 and variance uses ddof=1.
5. range is maximum minus minimum.
6. mode is the most frequent value requested for each applicable column.
7. Preserve column names and categorical values exactly as spoken.
8. allowed_values must preserve any order specified in the transcript.
9. value_range must use the exact structure requested in the transcript.
10. correlation must preserve the exact requested structure and order.
11. Use Pearson correlation unless another method is explicitly requested.
12. JSON numeric values must be numbers, not strings.
13. Empty objects must be {} and empty arrays must be [].
14. Do not add any extra top-level keys.
15. Check all arithmetic carefully.
16. Return only valid JSON, with no markdown or explanation.
""".strip()

    user_prompt = f"""
Audio ID: {audio_id}

Korean transcript:
{transcript}

Analyze the described dataset and return the required JSON.
""".strip()

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={
                "type": "json_object",
            },
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = completion.choices[0].message.content

        if not content:
            raise HTTPException(
                status_code=500,
                detail="Statistics model returned an empty response",
            )

        result = clean_json_response(content)
        return validate_result(result)

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"Q6 analysis error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {str(error)}",
        ) from error


@router.post(
    "/audio-stats",
    response_model=AudioStatisticsResponse,
)
def audio_statistics(request: AudioStatisticsRequest):
    audio_bytes = decode_audio(request.audio_base64)

    print(
        f"Q6 request: audio_id={request.audio_id}, "
        f"bytes={len(audio_bytes)}, "
        f"format={detect_audio_extension(audio_bytes)}",
        flush=True,
    )

    transcript = transcribe_audio(audio_bytes)

    print(
        f"Q6 transcript for {request.audio_id}: {transcript}",
        flush=True,
    )

    result = analyze_transcript(
        audio_id=request.audio_id,
        transcript=transcript,
    )

    return result


@router.get("/audio-stats")
def audio_statistics_information():
    return {
        "message": "Use POST with audio_id and audio_base64"
    }
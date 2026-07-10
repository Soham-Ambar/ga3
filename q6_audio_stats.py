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

    # Also accept data URLs such as:
    # data:audio/wav;base64,UklGR...
    if encoded.lower().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    # Remove spaces and line breaks from base64.
    encoded = "".join(encoded.split())

    try:
        audio_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
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
    # WAV: RIFF....WAVE
    if (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    ):
        return ".wav"

    # MP3 containing an ID3 header.
    if audio_bytes[:3] == b"ID3":
        return ".mp3"

    # MP3 frame header.
    if len(audio_bytes) >= 2:
        first_byte = audio_bytes[0]
        second_byte = audio_bytes[1]

        if first_byte == 0xFF and (second_byte & 0xE0) == 0xE0:
            return ".mp3"

    # FLAC.
    if audio_bytes[:4] == b"fLaC":
        return ".flac"

    # OGG.
    if audio_bytes[:4] == b"OggS":
        return ".ogg"

    # WebM or Matroska.
    if audio_bytes[:4] == bytes.fromhex("1A45DFA3"):
        return ".webm"

    # MP4 or M4A container.
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".m4a"

    # The assignment normally provides WAV audio.
    return ".wav"


def transcribe_audio(audio_bytes: bytes) -> str:
    client = get_groq_client()
    extension = detect_audio_extension(audio_bytes)

    temporary_path: str | None = None

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

        transcript = getattr(
            transcription,
            "text",
            None,
        )

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
            "Q6 transcription error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Audio transcription failed: {error}",
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
    elif cleaned.startswith("```JSON"):
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
            detail="Statistics model did not return a JSON object",
        )

    return result


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    missing_keys = set(REQUIRED_KEYS) - set(result.keys())

    if missing_keys:
        raise HTTPException(
            status_code=500,
            detail=f"Missing result keys: {sorted(missing_keys)}",
        )

    # Rebuild the response to remove all unexpected top-level keys.
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
        validated = AudioStatisticsResponse.model_validate(
            normalized
        )
    except Exception as error:
        print(
            f"Q6 validation error: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Incorrect result data types: {error}",
        ) from error

    return validated.model_dump()


def analyze_transcript(
    audio_id: str,
    transcript: str,
) -> dict[str, Any]:
    client = get_groq_client()

    system_prompt = """
You are a precise Korean-language dataset instruction parser and calculator.

The Korean audio transcript describes a dataset and explicitly requests
particular outputs. Return only values explicitly requested in the audio.

You must always return exactly these thirteen top-level keys:

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

CRITICAL RULE:

Do not automatically calculate every statistic for every numeric column.

For each statistics section, include a column only when the Korean audio
explicitly requests that exact statistic for that column.

Examples:

If the audio requests only the mean of "소득", return:

{
  "rows": 0,
  "columns": [],
  "mean": {"소득": 123},
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

If the audio requests minimum but not maximum, populate "min" and leave
"max" as {}.

If it requests maximum but not minimum, populate "max" and leave "min"
as {}.

If it requests variance but not standard deviation, populate "variance"
and leave "std" as {}.

If it requests standard deviation but not variance, populate "std" and
leave "variance" as {}.

If it requests allowed values but not mode, populate "allowed_values"
and leave "mode" as {}.

If no correlation is requested, return "correlation": [].

Never infer that a related statistic was requested:

- Requesting min does not imply max.
- Requesting max does not imply min.
- Requesting min and max does not imply range.
- Requesting variance does not imply standard deviation.
- Requesting standard deviation does not imply variance.
- Requesting mean does not imply median.
- Requesting mean does not imply mode.
- Requesting allowed values does not imply mode.
- Requesting observed minimum and maximum does not imply value_range.

Detailed rules:

1. "rows":
   Return the dataset row count only when the audio explicitly requests
   or specifies that rows must be returned. Otherwise return 0.

2. "columns":
   Return column names only when the audio explicitly requests or specifies
   them as an output. Preserve their exact required order.
   Otherwise return [].

3. For mean, std, variance, min, max, median, mode, and range:
   populate only the exact requested section and the exact requested columns.

4. Use sample standard deviation and sample variance compatible with pandas:
   Series.std(ddof=1)
   Series.var(ddof=1)

5. Calculate range as maximum minus minimum only when "range" itself is
   explicitly requested.

6. "allowed_values":
   Include only explicitly requested categorical allowed values.
   Preserve their exact order when an order is stated.

7. "value_range":
   Include only explicitly requested permitted ranges, bounds, or domains.
   Do not populate value_range merely because min and max can be computed.

8. "correlation":
   Include correlations only when explicitly requested.
   Use Pearson correlation unless another method is explicitly stated.
   Preserve the required structure and order exactly.

9. Preserve Korean column names and categorical values exactly as required
   by the transcript.

10. JSON numbers must be numbers, not strings.

11. Every unrequested dictionary section must be exactly {}.

12. Every unrequested array section must be exactly [].

13. Do not add extra top-level keys.

14. Do not populate fields merely because the necessary data is available.

15. Before returning, inspect every non-empty field and confirm that the
    transcript explicitly requested it. Remove every unrequested value.

16. Check all arithmetic carefully.

Return only one valid JSON object.
Do not return markdown, commentary, translation, or explanation.
""".strip()

    user_prompt = f"""
Audio ID: {audio_id}

Korean transcript:

--- START OF TRANSCRIPT ---
{transcript}
--- END OF TRANSCRIPT ---

First determine exactly which output fields and statistics the Korean audio
requests.

Then calculate only those requested outputs.

Populate only explicitly requested fields.

Leave every unrequested dictionary as {{}}.
Leave every unrequested list as [].
Use 0 for rows when rows are not requested.

Do not calculate or return additional related statistics.

Return only the required JSON object.
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
            "Q6 analysis error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {error}",
        ) from error


@router.post(
    "/audio-stats",
    response_model=AudioStatisticsResponse,
)
def audio_statistics(
    request: AudioStatisticsRequest,
) -> dict[str, Any]:
    audio_bytes = decode_audio(request.audio_base64)

    detected_format = detect_audio_extension(audio_bytes)

    print(
        f"Q6 request: audio_id={request.audio_id}, "
        f"bytes={len(audio_bytes)}, "
        f"format={detected_format}",
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

    print(
        "Q6 final response: "
        + json.dumps(
            result,
            ensure_ascii=False,
        ),
        flush=True,
    )

    return result


@router.get("/audio-stats")
def audio_statistics_information():
    return {
        "message": "Use POST with audio_id and audio_base64"
    }
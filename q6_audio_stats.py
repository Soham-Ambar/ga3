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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# These are the only top-level keys the grader allows.
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


def decode_audio(audio_base64: str) -> bytes:
    """
    Convert the base64 string sent by the grader into raw audio bytes.

    It also supports data-URL input such as:
    data:audio/wav;base64,UklGR...
    """

    encoded_data = audio_base64.strip()

    if "," in encoded_data and encoded_data.lower().startswith("data:"):
        encoded_data = encoded_data.split(",", 1)[1]

    # Remove whitespace or newlines that may exist in the base64 string.
    encoded_data = "".join(encoded_data.split())

    try:
        return base64.b64decode(encoded_data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="audio_base64 is not valid base64 audio",
        ) from error


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Send the decoded audio to Groq Whisper and return its transcript.
    """

    temporary_path = None

    try:
        # The exact incoming audio format may be WAV, MP3, M4A, etc.
        # Groq detects the real format from the file contents.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".audio",
        ) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        with open(temporary_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(f"recording_{os.path.basename(temporary_path)}", audio_file),
                model="whisper-large-v3",
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
                detail="The audio could not be transcribed",
            )

        return transcript.strip()

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Audio transcription failed: {str(error)}",
        ) from error

    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def parse_json_content(content: str) -> dict[str, Any]:
    """
    Parse JSON returned by the language model.

    This also removes accidental ```json fences if the model adds them.
    """

    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json")
        cleaned = cleaned.removeprefix("```JSON")
        cleaned = cleaned.removeprefix("```")
        cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=502,
            detail="The statistics model returned invalid JSON",
        ) from error

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail="The statistics model did not return a JSON object",
        )

    return parsed


def validate_and_normalize(result: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure the response contains exactly the keys required by the grader.
    """

    result_keys = set(result.keys())
    expected_keys = set(REQUIRED_KEYS)

    missing = expected_keys - result_keys

    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Statistics response is missing keys: {sorted(missing)}",
        )

    # Construct a brand-new object so that accidental extra keys are removed.
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
        raise HTTPException(
            status_code=502,
            detail=f"Statistics response has incorrect data types: {str(error)}",
        ) from error

    return validated.model_dump()


def analyze_transcript(
    audio_id: str,
    transcript: str,
) -> dict[str, Any]:
    """
    Ask the language model to understand the Korean dataset description
    and calculate the requested statistics.
    """

    system_prompt = """
You are a highly accurate dataset-statistics extraction engine.

You will receive a transcript of Korean speech. The speech describes a
dataset, table, records, columns, values, categories, ranges, and statistics.

Your job is to understand the Korean transcript and return the exact dataset
statistics requested by the speaker.

Important calculation rules:

1. Translate and understand Korean accurately, but preserve column names and
   category values exactly as specified in the audio.

2. "rows" is the number of data rows, excluding any header.

3. "columns" must contain column names in their original stated order.

4. For every numeric column, calculate:
   - mean
   - standard deviation
   - variance
   - minimum
   - maximum
   - median
   - mode
   - range

5. Use sample standard deviation and sample variance, equivalent to:
   pandas Series.std(ddof=1)
   pandas Series.var(ddof=1)

6. range means maximum minus minimum.

7. For categorical columns, use allowed_values where requested. Preserve the
   stated value order when the transcript specifies an order.

8. value_range contains the minimum and maximum permitted or observed limits
   requested in the speech.

9. correlation must follow the exact format and column order requested in the
   speech. Use Pearson correlation when correlation is requested.

10. Do not invent data that is not present in the transcript.

11. Numbers must be JSON numbers, not strings.

12. Empty sections must be represented by {} or [] as appropriate.

13. Return exactly these thirteen top-level keys and no others:
    rows
    columns
    mean
    std
    variance
    min
    max
    median
    mode
    range
    allowed_values
    value_range
    correlation

Required JSON shape:

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

Check every arithmetic result carefully before returning the answer.
Return only one valid JSON object. Do not return markdown or explanations.
""".strip()

    user_prompt = f"""
Audio identifier: {audio_id}

Korean audio transcript:

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Extract the dataset and calculate all requested statistics. Return only the
required JSON object.
""".strip()

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
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
                status_code=502,
                detail="The statistics model returned an empty response",
            )

        result = parse_json_content(content)
        return validate_and_normalize(result)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Dataset analysis failed: {str(error)}",
        ) from error


@router.post(
    "/audio-stats",
    response_model=AudioStatisticsResponse,
    response_model_exclude_none=False,
)
def audio_statistics(request: AudioStatisticsRequest):
    """
    Q6 endpoint:

    1. Decode the base64 audio.
    2. Transcribe the Korean speech.
    3. Extract and calculate the dataset statistics.
    4. Return exactly the required JSON structure.
    """

    audio_bytes = decode_audio(request.audio_base64)

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Decoded audio is empty",
        )

    transcript = transcribe_audio(audio_bytes)

    return analyze_transcript(
        audio_id=request.audio_id,
        transcript=transcript,
    )
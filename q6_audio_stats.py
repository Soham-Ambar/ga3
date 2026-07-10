import base64
import binascii
import json
import math
import os
import statistics
import tempfile
from collections import Counter
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


STATISTIC_KEYS = [
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "median",
    "mode",
    "range",
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

    if encoded.lower().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

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
    if (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    ):
        return ".wav"

    if audio_bytes[:3] == b"ID3":
        return ".mp3"

    if len(audio_bytes) >= 2:
        if (
            audio_bytes[0] == 0xFF
            and (audio_bytes[1] & 0xE0) == 0xE0
        ):
            return ".mp3"

    if audio_bytes[:4] == b"fLaC":
        return ".flac"

    if audio_bytes[:4] == b"OggS":
        return ".ogg"

    if audio_bytes[:4] == bytes.fromhex("1A45DFA3"):
        return ".webm"

    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".m4a"

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
            detail="Parser model returned invalid JSON",
        ) from error

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Parser model did not return a JSON object",
        )

    return result


def parse_transcript(
    audio_id: str,
    transcript: str,
) -> dict[str, Any]:
    """
    Ask the LLM only to extract structure.

    It does not calculate statistics. Python calculates them later.
    """

    client = get_groq_client()

    system_prompt = """
You are a Korean dataset-instruction parser.

The Korean transcript describes a small dataset and says which outputs must
be returned.

Your task is only to extract the dataset and requested operations.
Do not calculate statistics.

Return exactly one JSON object with this structure:

{
  "columns": [],
  "data": {},
  "requested": {
    "rows": false,
    "columns": false,
    "mean": [],
    "std": [],
    "variance": [],
    "min": [],
    "max": [],
    "median": [],
    "mode": [],
    "range": [],
    "allowed_values": [],
    "value_range": [],
    "correlation": []
  },
  "explicit_allowed_values": {},
  "explicit_value_range": {}
}

Rules:

1. "columns":
   Include every dataset column described in the audio.
   Preserve the exact Korean spelling and dataset order.

2. "data":
   Map each column name to its values in row order.

   Example:
   {
     "나이": [20, 25, 30],
     "도시": ["서울", "부산", "서울"]
   }

3. Convert spoken Korean numbers to JSON numbers.

4. Keep categorical values as exact strings.

5. "requested.rows":
   true only when row count is requested as an output.

6. "requested.columns":
   true only when column names are requested as an output.

7. For each statistic:
   list only columns for which that exact statistic is requested.

8. Never infer related statistics:
   - mean does not imply median
   - min does not imply max
   - min plus max does not imply range
   - variance does not imply std
   - std does not imply variance
   - mode does not imply allowed_values
   - min/max does not imply value_range

9. "requested.allowed_values":
   list columns whose allowed values are explicitly requested.

10. "requested.value_range":
    list columns whose permitted or valid range is explicitly requested.

11. "requested.correlation":
    use an array of objects:
    [
      {
        "column1": "키",
        "column2": "몸무게"
      }
    ]

12. "explicit_allowed_values":
    when the audio explicitly states allowed values, place them here.

    Example:
    {
      "성별": ["남성", "여성"]
    }

13. "explicit_value_range":
    when the audio explicitly states a permitted range, place it here.

    Example:
    {
      "나이": [0, 100]
    }

14. If allowed values or ranges must be derived from the data instead of
    being explicitly stated, leave their explicit objects empty.

15. Return only valid JSON.

16. Do not calculate mean, variance, correlation or any other statistic.
""".strip()

    user_prompt = f"""
Audio ID: {audio_id}

Korean transcript:

--- START ---
{transcript}
--- END ---

Extract the complete dataset, all columns in order, all row values, and the
exact requested output operations.

Return only the parser JSON object.
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
                detail="Parser model returned an empty response",
            )

        parsed = clean_json_response(content)

        print(
            "Q6 parsed structure: "
            + json.dumps(
                parsed,
                ensure_ascii=False,
            ),
            flush=True,
        )

        return parsed

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Q6 parser error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Transcript parsing failed: {error}",
        ) from error


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(float(value))
    )


def numeric_values(
    data: dict[str, Any],
    column: str,
) -> list[float | int]:
    values = data.get(column, [])

    if not isinstance(values, list):
        return []

    return [
        value
        for value in values
        if is_number(value)
    ]


def normalize_number(value: float | int) -> float | int:
    if isinstance(value, bool):
        return int(value)

    numeric = float(value)

    if math.isclose(
        numeric,
        round(numeric),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        return int(round(numeric))

    return numeric


def calculate_mean(values: list[float | int]) -> float | int:
    return normalize_number(
        sum(values) / len(values)
    )


def calculate_sample_variance(
    values: list[float | int],
) -> float | int | None:
    if len(values) < 2:
        return None

    mean_value = sum(values) / len(values)

    variance_value = sum(
        (value - mean_value) ** 2
        for value in values
    ) / (len(values) - 1)

    return normalize_number(variance_value)


def calculate_sample_std(
    values: list[float | int],
) -> float | int | None:
    variance_value = calculate_sample_variance(values)

    if variance_value is None:
        return None

    return normalize_number(
        math.sqrt(float(variance_value))
    )


def calculate_median(
    values: list[float | int],
) -> float | int:
    return normalize_number(
        statistics.median(values)
    )


def calculate_mode(values: list[Any]) -> Any:
    if not values:
        return None

    counts = Counter(values)
    highest_count = max(counts.values())

    # Match pandas-style first mode in sorted order when possible.
    modes = [
        value
        for value, count in counts.items()
        if count == highest_count
    ]

    try:
        modes = sorted(modes)
    except TypeError:
        pass

    return modes[0]


def calculate_pearson(
    first: list[float | int],
    second: list[float | int],
) -> float | int | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(first, second)
        if is_number(x) and is_number(y)
    ]

    if len(pairs) < 2:
        return None

    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in pairs
    )

    x_squared = sum(
        (x - x_mean) ** 2
        for x in x_values
    )

    y_squared = sum(
        (y - y_mean) ** 2
        for y in y_values
    )

    denominator = math.sqrt(
        x_squared * y_squared
    )

    if denominator == 0:
        return None

    return normalize_number(
        numerator / denominator
    )


def get_unique_in_order(values: list[Any]) -> list[Any]:
    unique_values = []

    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    return unique_values


def get_requested_columns(
    requested: dict[str, Any],
    key: str,
) -> list[str]:
    value = requested.get(key, [])

    if not isinstance(value, list):
        return []

    return [
        str(column)
        for column in value
    ]


def build_response(
    parsed: dict[str, Any],
) -> dict[str, Any]:
    columns = parsed.get("columns", [])
    data = parsed.get("data", {})
    requested = parsed.get("requested", {})
    explicit_allowed_values = parsed.get(
        "explicit_allowed_values",
        {},
    )
    explicit_value_range = parsed.get(
        "explicit_value_range",
        {},
    )

    if not isinstance(columns, list):
        columns = []

    if not isinstance(data, dict):
        data = {}

    if not isinstance(requested, dict):
        requested = {}

    if not isinstance(explicit_allowed_values, dict):
        explicit_allowed_values = {}

    if not isinstance(explicit_value_range, dict):
        explicit_value_range = {}

    result: dict[str, Any] = {
        "rows": 0,
        "columns": columns,
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
        "correlation": [],
    }

    # Determine number of rows from the longest data column.
    row_count = 0

    for column in columns:
        values = data.get(column, [])

        if isinstance(values, list):
            row_count = max(
                row_count,
                len(values),
            )

    # The assignment examples indicate rows and columns describe the dataset.
    # Therefore, always return them when a dataset exists.
    result["rows"] = row_count
    result["columns"] = columns

    for column in get_requested_columns(
        requested,
        "mean",
    ):
        values = numeric_values(data, column)

        if values:
            result["mean"][column] = calculate_mean(values)

    for column in get_requested_columns(
        requested,
        "std",
    ):
        values = numeric_values(data, column)
        value = calculate_sample_std(values)

        if value is not None:
            result["std"][column] = value

    for column in get_requested_columns(
        requested,
        "variance",
    ):
        values = numeric_values(data, column)
        value = calculate_sample_variance(values)

        if value is not None:
            result["variance"][column] = value

    for column in get_requested_columns(
        requested,
        "min",
    ):
        values = data.get(column, [])

        if isinstance(values, list) and values:
            try:
                result["min"][column] = min(values)
            except TypeError:
                numeric = numeric_values(data, column)

                if numeric:
                    result["min"][column] = min(numeric)

    for column in get_requested_columns(
        requested,
        "max",
    ):
        values = data.get(column, [])

        if isinstance(values, list) and values:
            try:
                result["max"][column] = max(values)
            except TypeError:
                numeric = numeric_values(data, column)

                if numeric:
                    result["max"][column] = max(numeric)

    for column in get_requested_columns(
        requested,
        "median",
    ):
        values = numeric_values(data, column)

        if values:
            result["median"][column] = calculate_median(values)

    for column in get_requested_columns(
        requested,
        "mode",
    ):
        values = data.get(column, [])

        if isinstance(values, list) and values:
            mode_value = calculate_mode(values)

            if mode_value is not None:
                result["mode"][column] = mode_value

    for column in get_requested_columns(
        requested,
        "range",
    ):
        values = numeric_values(data, column)

        if values:
            result["range"][column] = normalize_number(
                max(values) - min(values)
            )

    for column in get_requested_columns(
        requested,
        "allowed_values",
    ):
        if column in explicit_allowed_values:
            values = explicit_allowed_values[column]

            if isinstance(values, list):
                result["allowed_values"][column] = values
        else:
            values = data.get(column, [])

            if isinstance(values, list):
                result["allowed_values"][column] = (
                    get_unique_in_order(values)
                )

    for column in get_requested_columns(
        requested,
        "value_range",
    ):
        if column in explicit_value_range:
            result["value_range"][column] = (
                explicit_value_range[column]
            )
        else:
            values = numeric_values(data, column)

            if values:
                result["value_range"][column] = [
                    normalize_number(min(values)),
                    normalize_number(max(values)),
                ]

    correlation_requests = requested.get(
        "correlation",
        [],
    )

    if isinstance(correlation_requests, list):
        for request in correlation_requests:
            if not isinstance(request, dict):
                continue

            column1 = request.get("column1")
            column2 = request.get("column2")

            if not column1 or not column2:
                continue

            first_values = numeric_values(
                data,
                str(column1),
            )

            second_values = numeric_values(
                data,
                str(column2),
            )

            correlation_value = calculate_pearson(
                first_values,
                second_values,
            )

            if correlation_value is not None:
                result["correlation"].append(
                    {
                        "column1": str(column1),
                        "column2": str(column2),
                        "correlation": correlation_value,
                    }
                )

    return result


def validate_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    missing_keys = set(REQUIRED_KEYS) - set(result.keys())

    if missing_keys:
        raise HTTPException(
            status_code=500,
            detail=f"Missing result keys: {sorted(missing_keys)}",
        )

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

    parsed = parse_transcript(
        audio_id=request.audio_id,
        transcript=transcript,
    )

    result = build_response(parsed)
    result = validate_result(result)

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
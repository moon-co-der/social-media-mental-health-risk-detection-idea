"""FastAPI inference service for the research demonstration."""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_VERSION = "0.1.0"
E2_MODEL_PATH = os.getenv("E2_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "e2" / "best_model"))
E2_HF_MODEL_ID = os.getenv("E2_HF_MODEL_ID", "")
E2_MODEL_REVISION = os.getenv("E2_MODEL_REVISION", "")
EMOTION_MODEL_ID = os.getenv("EMOTION_MODEL_ID", "SamLowe/roberta-base-go_emotions")
EMOTION_MODEL_REVISION = os.getenv("EMOTION_MODEL_REVISION", "906933dcd4245e4f48d8a2ed585b036e2dd45f80")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))
BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
if min(MAX_UPLOAD_BYTES, MAX_ROWS, BATCH_SIZE) <= 0:
    raise ValueError("MAX_UPLOAD_BYTES, MAX_ROWS, and INFERENCE_BATCH_SIZE must all be positive")
MAX_LENGTH = 512
EMOTION_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)
TEMPORAL = {"recent_days": 90, "baseline_days": 180, "min_recent_posts": 5, "min_baseline_posts": 15,
            "recent_age_interval_days": "[0, 90)", "baseline_age_interval_days": "[90, 270)",
            "anchor": "latest valid timestamp per user; UTC"}


class RequestTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Cap multipart body size before FastAPI spools uploaded files to disk."""
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in {"/analyze", "/validate"} or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        limit = self.max_bytes + 65536  # room for multipart boundaries and mapped form fields
        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > limit:
                return await self._reject(send)
        except ValueError:
            return await self._reject(send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLarge:
            await self._reject(send)

    async def _reject(self, send):
        body = b'{"detail":"Request exceeds the configured upload limit. No data was processed."}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


app = FastAPI(title="Social Media NLP Research API", version=APP_VERSION)
logger = logging.getLogger("nlp-inference")
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], allow_credentials=False)

_models: dict[str, Any] = {}
_model_error: str | None = None
_model_load_seconds: float | None = None
_model_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_KEYS = {"e2_tokenizer", "e2_model", "emo_tokenizer", "emo_model", "e2_labels", "emotions"}


def _models_ready() -> bool:
    return _MODEL_KEYS.issubset(_models)


def _load_and_infer(texts: list[str]) -> tuple[list[Any], ...]:
    models = _load_models()
    with _inference_lock:
        return _infer(texts, models)


def _load_models() -> dict[str, Any]:
    global _model_error, _model_load_seconds
    if _models_ready():
        return _models
    with _model_load_lock:
        if _models_ready():
            return _models
        load_started = time.perf_counter()
        try:
            if "e2_model" not in _models:
                e2_path = Path(E2_MODEL_PATH)
                is_local_e2 = e2_path.is_dir()
                is_hub_e2 = bool(E2_HF_MODEL_ID)
                if not is_local_e2 and not is_hub_e2:
                    raise FileNotFoundError(f"E2 checkpoint not found or directory invalid: {e2_path}. Set E2_MODEL_PATH to the extracted best_model directory.")
                if is_local_e2:
                    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
                    missing = [name for name in required if not (e2_path / name).is_file()]
                    if missing:
                        raise FileNotFoundError(f"E2 checkpoint is incomplete at {e2_path}; missing: {', '.join(missing)}")
                e2_kwargs = {"local_files_only": True} if is_local_e2 else {"token": os.getenv("HF_TOKEN") or None}
                if E2_MODEL_REVISION and is_hub_e2:
                    e2_kwargs["revision"] = E2_MODEL_REVISION
                e2_source = str(e2_path) if is_local_e2 else E2_HF_MODEL_ID
                e2_tokenizer = AutoTokenizer.from_pretrained(e2_source, **e2_kwargs)
                e2_model = AutoModelForSequenceClassification.from_pretrained(e2_source, **e2_kwargs)
                expected_e2 = {0: "attempt", 1: "behavior", 2: "ideation", 3: "indicator", 4: "supportive", 5: "uninformative"}
                e2_labels = {int(k): str(v).strip().lower() for k, v in e2_model.config.id2label.items()}
                if e2_labels != expected_e2 or int(e2_model.config.num_labels) != 6:
                    raise ValueError(f"Unexpected E2 label mapping: {e2_labels}; expected {expected_e2}")
                e2_model.to(_device).eval()
                _models.update(e2_tokenizer=e2_tokenizer, e2_model=e2_model, e2_labels=e2_labels)
            emotion_kwargs = {"revision": EMOTION_MODEL_REVISION, "token": os.getenv("HF_TOKEN") or None}
            emo_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID, **emotion_kwargs)
            emo_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID, **emotion_kwargs)
            emotions = {int(k): str(v).strip().lower() for k, v in emo_model.config.id2label.items()}
            actual_emotions = tuple(emotions.get(i) for i in range(28))
            if int(emo_model.config.num_labels) != 28 or actual_emotions != EMOTION_LABELS:
                raise ValueError(f"GoEmotions label mapping differs from the pinned 28-label mapping: {actual_emotions}")
            emo_model.to(_device).eval()
            _models.update(emo_tokenizer=emo_tokenizer, emo_model=emo_model, emotions=emotions)
            _model_load_seconds = time.perf_counter() - load_started
            _model_error = None
            return _models
        except Exception as exc:
            logger.exception("Model loading failed")
            _model_error = str(exc)
            raise


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "nlp-inference", "version": APP_VERSION,
            "models_loaded": _models_ready(), "inference_available": _models_ready(),
            "model_error": "Model loading failed; check server configuration and host logs." if _model_error else None,
            "e2_checkpoint_configured": Path(E2_MODEL_PATH).exists() or bool(E2_HF_MODEL_ID),
            "emotion_model": EMOTION_MODEL_ID,
            "emotion_revision_pinned": bool(EMOTION_MODEL_REVISION), "device": str(_device), "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_rows": MAX_ROWS}


def parse_csv(data: bytes, text_column: str, user_column: str, timestamp_column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not data:
        raise ValueError("The uploaded CSV is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if len(frame.columns) == 0:
        raise ValueError("The CSV has no header row.")
    if len(frame) == 0:
        raise ValueError("The CSV contains a header but no data rows.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"This service currently accepts up to {MAX_ROWS} rows; the file has {len(frame)}. No rows were truncated.")
    if text_column not in frame.columns:
        raise ValueError(f"Text column '{text_column}' does not exist. Available columns: {', '.join(frame.columns)}")
    for col in (user_column, timestamp_column):
        if col and col not in frame.columns:
            raise ValueError(f"Mapped column '{col}' does not exist.")
    frame = frame.copy()
    frame["_text"] = frame[text_column].astype(str)
    frame["_text_valid"] = frame["_text"].str.strip().ne("")
    frame["_user"] = frame[user_column].astype(str).str.strip() if user_column else ""
    raw_time = frame[timestamp_column].astype(str).str.strip() if timestamp_column else pd.Series("", index=frame.index)
    # Numeric epochs are deliberately rejected; pandas otherwise reads them as nanoseconds.
    numeric = raw_time.ne("") & raw_time.str.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)")
    parsed = pd.to_datetime(raw_time.mask(numeric), errors="coerce", utc=True, format="mixed")
    frame["_timestamp"] = parsed
    frame["_numeric_timestamp"] = numeric
    valid_temporal = bool(user_column and timestamp_column)
    valid_mask = frame["_text_valid"] & frame["_user"].ne("") & frame["_timestamp"].notna() if valid_temporal else pd.Series(False, index=frame.index)
    duplicate_rows = int(frame.drop(columns=["_text", "_text_valid", "_user", "_timestamp", "_numeric_timestamp"]).duplicated(keep="first").sum())
    times = frame.loc[frame["_text_valid"] & frame["_timestamp"].notna(), "_timestamp"]
    valid_text = frame.loc[frame["_text_valid"], "_text"]
    exact_text_duplicates = int(valid_text.duplicated(keep="first").sum())
    normalized_text = valid_text.map(lambda value: re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", value).casefold(), flags=re.UNICODE))
    normalized_duplicates = int(normalized_text[normalized_text.ne("")].duplicated(keep="first").sum())
    info = {"rows_received": int(len(frame)), "nonempty_text_rows": int(frame["_text_valid"].sum()),
            "empty_text_rows": int((~frame["_text_valid"]).sum()), "duplicate_rows": duplicate_rows,
            "exact_duplicate_text_rows": exact_text_duplicates,
            "normalized_near_duplicate_text_rows": max(0, normalized_duplicates - exact_text_duplicates),
            "very_short_text_rows_under_5_words": int(valid_text.map(lambda value: len(value.split()) < 5).sum()),
            "mapped_columns": {"text": bool(text_column), "user_id": bool(user_column), "timestamp": bool(timestamp_column)},
            "unique_users": int(frame.loc[frame["_user"].ne(""), "_user"].nunique()) if user_column else 0,
            "missing_user_id_rows": int(frame["_user"].eq("").sum()) if user_column else int(len(frame)),
            "missing_or_invalid_timestamp_rows": int(frame["_timestamp"].isna().sum()),
            "numeric_timestamp_rows": int(numeric.sum()), "temporal_analysis_available": valid_temporal,
            "parsed_timestamp_min": times.min().isoformat() if len(times) else None,
            "parsed_timestamp_max": times.max().isoformat() if len(times) else None}
    if valid_temporal:
        eligible = _eligibility(frame.loc[valid_mask])
        info["temporal_eligible_users"] = sum(u["eligible"] for u in eligible)
        info["temporal_ineligible_users"] = len(eligible) - info["temporal_eligible_users"]
    else:
        info["temporal_eligible_users"] = 0
        info["temporal_ineligible_users"] = info["unique_users"]
    return frame, info


def validate_upload_size(contents: bytes) -> None:
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MiB upload limit. No data was processed.")


def validation_details(frame: pd.DataFrame, summary: dict[str, Any], pseudonym_seed: str | None = None) -> dict[str, Any]:
    eligible_rows = frame.loc[frame["_text_valid"] & frame["_user"].ne("") & frame["_timestamp"].notna()]
    users = _eligibility(eligible_rows) if summary["temporal_analysis_available"] else []
    pseudonym_seed = pseudonym_seed or hashlib.sha256(os.urandom(32)).hexdigest()
    salt = pseudonym_seed.encode()
    public_users = [{"user_id": "user_" + hashlib.sha256(salt + item["raw_user"].encode()).hexdigest()[:10],
                     "recent_posts": item["recent_posts"], "baseline_posts": item["baseline_posts"],
                     "eligible": item["eligible"], "reason": item["reason"]} for item in users]
    return {"summary": summary, "users": public_users, "pseudonym_seed": pseudonym_seed}


@app.post("/validate")
async def validate(file: UploadFile = File(...), text_column: str = Form(...), user_column: str = Form(""), timestamp_column: str = Form("")) -> JSONResponse:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    validate_upload_size(contents)
    try:
        frame, summary = parse_csv(contents, text_column, user_column, timestamp_column)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(validation_details(frame, summary), headers={"Cache-Control": "no-store"})


def _eligibility(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for user, group in frame.groupby("_user", sort=True):
        anchor = group["_timestamp"].max()
        age = (anchor - group["_timestamp"]).dt.total_seconds() / 86400
        recent = int(((age >= 0) & (age < 90)).sum())
        baseline = int(((age >= 90) & (age < 270)).sum())
        eligible = baseline >= 15 and recent >= 5
        reasons = []
        if recent < 5: reasons.append(f"{recent} recent posts; 5 required")
        if baseline < 15: reasons.append(f"{baseline} baseline posts; 15 required")
        rows.append({"raw_user": user, "anchor": anchor, "recent_posts": recent, "baseline_posts": baseline,
                     "eligible": eligible, "reason": "Eligible" if eligible else "; ".join(reasons)})
    return rows


def _token_counts(texts: list[str], tokenizer: Any) -> list[int]:
    """Count each model's full input tokens, including special tokens, without returning text."""
    return [len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]


def _infer(texts: list[str], models: dict[str, Any]) -> tuple[list[Any], ...]:
    classes, scores, e2_probabilities, emotions = [], [], [], []
    started = time.perf_counter()
    e2 = models["e2_model"]
    e2_token_counts = _token_counts(texts, models["e2_tokenizer"])
    emotion_token_counts = _token_counts(texts, models["emo_tokenizer"])
    label_to_id = {v: k for k, v in models["e2_labels"].items()}
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        inputs = models["e2_tokenizer"](batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(_device)
        with torch.inference_mode(): probs = torch.softmax(e2(**inputs).logits, dim=-1).cpu().numpy()
        for row in probs:
            classes.append(models["e2_labels"][int(np.argmax(row))])
            scores.append(float(row[label_to_id["behavior"]] + row[label_to_id["attempt"]]))
            e2_probabilities.append({models["e2_labels"][i]: float(row[i]) for i in range(len(row))})
    emotion_model = models["emo_model"]
    for start in range(0, len(texts), BATCH_SIZE):
        inputs = models["emo_tokenizer"](texts[start:start+BATCH_SIZE], padding=True, truncation=True,
                                          max_length=MAX_LENGTH, return_tensors="pt").to(_device)
        with torch.inference_mode(): probs = torch.sigmoid(emotion_model(**inputs).logits).cpu().numpy()
        for row in probs:
            emotions.append({models["emotions"][i]: float(row[i]) for i in range(len(row))})
        return classes, scores, e2_probabilities, emotions, e2_token_counts, emotion_token_counts, time.perf_counter() - started


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), text_column: str = Form(...), user_column: str = Form(""), timestamp_column: str = Form(""), pseudonym_seed: str = Form("")) -> JSONResponse:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    validate_upload_size(contents)
    try:
        frame, summary = parse_csv(contents, text_column, user_column, timestamp_column)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    valid = frame["_text_valid"]
    if not bool(valid.any()):
        raise HTTPException(400, "No nonempty text rows remain for inference. Review the empty-text count from validation.")
    indices = frame.index[valid].tolist()
    try:
        load_was_needed = not _models_ready()
        classes, scores, e2_probabilities, emotions, e2_token_counts, emotion_token_counts, infer_seconds = await run_in_threadpool(
            _load_and_infer, frame.loc[valid, "_text"].tolist())
    except Exception as exc:
        detail = "The original E2 checkpoint is missing or incomplete. Configure E2_MODEL_PATH to its best_model directory." if isinstance(exc, FileNotFoundError) and "E2 checkpoint" in str(exc) else "Model loading or inference failed. Confirm both model artifacts, service memory, and Hugging Face download access; check host logs for details."
        raise HTTPException(503, detail) from exc
    prediction_by_index = {idx: j for j, idx in enumerate(indices)}
    temporal_data = []
    temporal_possible = summary["temporal_analysis_available"]
    if temporal_possible:
        temporal_data = _eligibility(frame.loc[valid & frame["_user"].ne("") & frame["_timestamp"].notna()])
    salt = pseudonym_seed.encode() if 1 <= len(pseudonym_seed) <= 128 else os.urandom(32)
    all_users = frame.loc[valid & frame["_user"].ne(""), "_user"].unique()
    user_alias = {user: "user_" + hashlib.sha256(salt + user.encode()).hexdigest()[:10] for user in all_users}
    predictions = []
    for idx in indices:
        j = prediction_by_index[idx]
        row = {"row_number": int(idx + 1), "predicted_class": classes[j], "e2_class_probabilities": e2_probabilities[j],
               "e2_score_uncalibrated": scores[j], "goemotions_scores": emotions[j],
               "e2_token_count": e2_token_counts[j], "goemotions_token_count": emotion_token_counts[j],
               "e2_truncated_at_512": e2_token_counts[j] > MAX_LENGTH,
               "goemotions_truncated_at_512": emotion_token_counts[j] > MAX_LENGTH}
        user = frame.at[idx, "_user"]
        row["user_id"] = user_alias.get(user)
        row["timestamp"] = frame.at[idx, "_timestamp"].isoformat() if pd.notna(frame.at[idx, "_timestamp"]) else None
        predictions.append(row)
    summary["e2_truncated_rows"] = sum(row["e2_truncated_at_512"] for row in predictions)
    summary["goemotions_truncated_rows"] = sum(row["goemotions_truncated_at_512"] for row in predictions)
    summary["truncated_by_either_model_rows"] = sum(row["e2_truncated_at_512"] or row["goemotions_truncated_at_512"] for row in predictions)
    users = []
    grouped = {}
    for idx in indices:
        user = frame.at[idx, "_user"]
        stamp = frame.at[idx, "_timestamp"]
        if user and pd.notna(stamp): grouped.setdefault(user, []).append(idx)
    temporal_lookup = {r["raw_user"]: r for r in temporal_data}
    for user, idxs in grouped.items():
        spec = temporal_lookup[user]
        recent_idx = [i for i in idxs if (spec["anchor"] - frame.at[i, "_timestamp"]).total_seconds()/86400 < 90]
        base_idx = [i for i in idxs if 90 <= (spec["anchor"] - frame.at[i, "_timestamp"]).total_seconds()/86400 < 270]
        def means(window: list[int], field: str) -> dict[str, float]:
            return {label: float(np.mean([emotions[prediction_by_index[i]][label] for i in window])) for label in emotions[0]} if window else {}
        er = [scores[prediction_by_index[i]] for i in recent_idx]
        eb = [scores[prediction_by_index[i]] for i in base_idx]
        recent_em = means(recent_idx, "emotion")
        baseline_em = means(base_idx, "emotion")
        users.append({"user_id": user_alias[user], "anchor": spec["anchor"].isoformat(), "eligible": spec["eligible"],
                      "reason": spec["reason"], "recent_posts": len(recent_idx), "baseline_posts": len(base_idx),
                      "recent_e2_mean": float(np.mean(er)) if er else None, "baseline_e2_mean": float(np.mean(eb)) if eb else None,
                      "e2_mean_difference_recent_minus_baseline": float(np.mean(er)-np.mean(eb)) if spec["eligible"] and er and eb else None,
                      "recent_emotion_means": recent_em, "baseline_emotion_means": baseline_em,
                      "emotion_differences_recent_minus_baseline": {k: recent_em[k]-baseline_em[k] for k in recent_em} if spec["eligible"] and recent_em and baseline_em else {}})
    return JSONResponse({"run": {"completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_inference_seconds": round(infer_seconds, 3),
                                  "model_loading_seconds": _model_load_seconds if load_was_needed else 0.0,
                                  "e2_model": "provided project E2 Weighted DistilBERT checkpoint",
                                  "e2_checkpoint_revision": "local artifact; original training revision not recorded",
                                  "e2_configured_revision": E2_MODEL_REVISION or "not pinned",
                                  "emotion_model": EMOTION_MODEL_ID,
                                  "emotion_revision": EMOTION_MODEL_REVISION,
                                  "device": str(_device), "max_length_tokens": MAX_LENGTH, "batch_size": BATCH_SIZE, "temporal": TEMPORAL,
                                  "warnings": ["Research demonstration only; not a diagnostic or clinical result.", "Scores are uncalibrated model outputs.",
                                               "GoEmotions values are independent sigmoid scores and do not sum to one.",
                                               "Rows may be truncated by the models at 512 tokens.",
                                               "Raw text is not returned or written to server-side result files."]},
                        "summary": summary, "predictions": predictions, "users": users},
                        headers={"Cache-Control": "no-store"})

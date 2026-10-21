import pandas as pd
import pytest
from fastapi import HTTPException

from pathlib import Path
from app.main import app, parse_csv, _eligibility, _load_models, _infer, validate_upload_size, validation_details



def csv_bytes(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode()


def temporal_rows():
    # 5 recent rows including anchor and 15 baseline rows within [90,270).
    rows = [{"body": f"recent {i}", "uid": "a", "when": f"2025-01-{i+1:02d}T00:00:00Z"} for i in range(5)]
    rows += [{"body": f"baseline {i}", "uid": "a", "when": f"2024-08-{i+1:02d}T00:00:00Z"} for i in range(15)]
    return rows


def test_empty_and_header_only_csv():
    with pytest.raises(ValueError, match="empty"):
        parse_csv(b"", "body", "uid", "when")
    with pytest.raises(ValueError, match="no data rows"):
        parse_csv(b"body,uid,when\n", "body", "uid", "when")


def test_missing_text_mapping():
    with pytest.raises(ValueError, match="does not exist"):
        parse_csv(csv_bytes([{"content": "hello"}]), "body", "", "")


def test_exact_thresholds_and_boundary_convention():
    rows = temporal_rows()
    rows.append({"body": "exactly 90 days", "uid": "a", "when": "2024-10-07T00:00:00Z"})
    rows.append({"body": "exactly 270 days", "uid": "a", "when": "2024-04-10T00:00:00Z"})
    _, summary = parse_csv(csv_bytes(rows), "body", "uid", "when")
    assert summary["temporal_eligible_users"] == 1
    assert summary["rows_received"] == 22
    frame, _ = parse_csv(csv_bytes(rows), "body", "uid", "when")
    eligibility = _eligibility(frame.loc[frame["_text_valid"]])
    # Anchor = Jan 5; exactly 90 days is baseline and exactly 270 is outside.
    assert eligibility[0]["recent_posts"] == 5
    assert eligibility[0]["baseline_posts"] == 16


def test_data_quality_text_counts_and_mapped_columns():
    frame, info = parse_csv(csv_bytes([
        {"body": "Same post here", "uid": "a", "when": "2025-01-01"},
        {"body": "Same post here", "uid": "b", "when": "2025-01-02"},
        {"body": "SAME, post here!", "uid": "c", "when": "2025-01-03"},
        {"body": "tiny words", "uid": "d", "when": "2025-01-04"},
        {"body": "   ", "uid": "e", "when": "2025-01-05"},
    ]), "body", "uid", "")
    assert info["exact_duplicate_text_rows"] == 1
    assert info["normalized_near_duplicate_text_rows"] == 1
    assert info["very_short_text_rows_under_5_words"] == 4
    assert info["empty_text_rows"] == 1
    assert info["mapped_columns"] == {"text": True, "user_id": True, "timestamp": False}


def test_full_token_counter_reports_per_model_counts():
    from app.main import _token_counts

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    assert _token_counts(["one two", "one two three"], Tokenizer()) == [2, 3]


def test_exact_minimum_counts_are_eligible():
    frame, summary = parse_csv(csv_bytes(temporal_rows()), "body", "uid", "when")
    assert summary["temporal_eligible_users"] == 1
    row = _eligibility(frame.loc[frame["_text_valid"]])[0]
    assert (row["recent_posts"], row["baseline_posts"]) == (5, 15)


def test_validation_summary_returns_pseudonymous_per_user_counts():
    frame, summary = parse_csv(csv_bytes(temporal_rows()), "body", "uid", "when")
    response = validation_details(frame, summary, "test-run-seed")
    assert response["summary"]["temporal_eligible_users"] == 1
    assert response["users"][0]["eligible"] is True
    assert response["users"][0]["user_id"].startswith("user_")
    assert response["users"][0]["user_id"] != "a"
    assert "raw_user" not in response["users"][0]
    assert validation_details(frame, summary, "test-run-seed")["users"] == response["users"]


def test_just_below_count_threshold_and_insufficient_history():
    rows = temporal_rows()[:-1]
    _, summary = parse_csv(csv_bytes(rows), "body", "uid", "when")
    assert summary["temporal_eligible_users"] == 0
    assert summary["temporal_ineligible_users"] == 1


def test_missing_user_and_timestamp_keep_post_rows():
    rows = [{"body": "one", "uid": "", "when": "2025-01-01"},
            {"body": "two", "uid": "a", "when": ""}]
    frame, info = parse_csv(csv_bytes(rows), "body", "uid", "when")
    assert info["nonempty_text_rows"] == 2
    assert info["temporal_eligible_users"] == 0
    assert frame["_text_valid"].all()


def test_bad_and_numeric_timestamps_are_not_epochs():
    rows = [{"body": "numeric", "uid": "u", "when": "1735689600"},
            {"body": "invalid", "uid": "u", "when": "not a date"},
            {"body": "naive", "uid": "u", "when": "2025-01-01 12:00:00"},
            {"body": "aware", "uid": "u", "when": "2025-01-02T12:00:00+05:30"}]
    _, info = parse_csv(csv_bytes(rows), "body", "uid", "when")
    assert info["numeric_timestamp_rows"] == 1
    assert info["missing_or_invalid_timestamp_rows"] == 2
    assert info["parsed_timestamp_min"].endswith("+00:00")


def test_whitespace_text_and_duplicates_are_counted_without_deduplication():
    rows = [{"body": "  ", "uid": "u", "when": "2025-01-01"},
            {"body": "same", "uid": "u", "when": "2025-01-02"},
            {"body": "same", "uid": "u", "when": "2025-01-02"}]
    frame, info = parse_csv(csv_bytes(rows), "body", "uid", "when")
    assert len(frame) == 3
    assert info["empty_text_rows"] == 1
    assert info["duplicate_rows"] == 1


def test_multiple_user_anchors_and_naive_timezone_utc():
    rows = temporal_rows()
    rows += [{"body": f"b recent {i}", "uid": "b", "when": f"2026-03-{i+1:02d} 10:00:00"} for i in range(5)]
    rows += [{"body": f"b baseline {i}", "uid": "b", "when": f"2025-10-{i+1:02d} 10:00:00+05:30"} for i in range(15)]
    frame, info = parse_csv(csv_bytes(rows), "body", "uid", "when")
    assert info["unique_users"] == 2
    assert info["temporal_eligible_users"] == 2
    assert str(frame.loc[0, "_timestamp"].tz) == "UTC"


def test_oversized_row_count_rejected(monkeypatch):
    import app.main as service
    monkeypatch.setattr(service, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="No rows were truncated"):
        parse_csv(csv_bytes([{"body": "a"}, {"body": "b"}]), "body", "", "")


def test_malformed_api_request():
    with pytest.raises(ValueError, match="Could not parse CSV"):
        parse_csv(b"\xff\xfe", "body", "", "")


def test_api_upload_byte_limit(monkeypatch):
    import app.main as service
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(HTTPException) as err:
        validate_upload_size(b"body\nhello")
    assert err.value.status_code == 413


def test_missing_e2_checkpoint_fails_clearly(monkeypatch):
    import app.main as service
    monkeypatch.setattr(service, "E2_MODEL_PATH", "models/definitely-not-present")
    service._models.clear()
    with pytest.raises(FileNotFoundError, match="E2 checkpoint not found"):
        _load_models()


def test_inference_failure_is_propagated_without_fallback():
    # Test-only doubles verify failure handling; production always loads the configured models.
    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

        def __call__(self, *args, **kwargs):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs()
    class FailingModel:
        def __call__(self, **kwargs):
            raise RuntimeError("deliberate test inference failure")
    models = {"e2_tokenizer": Tokenizer(), "emo_tokenizer": Tokenizer(), "e2_model": FailingModel(),
              "e2_labels": {0: "behavior", 1: "attempt"}}
    with pytest.raises(RuntimeError, match="deliberate test inference failure"):
        _infer(["test-only text"], models)


def test_real_e2_checkpoint_inference_smoke_when_configured():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import app.main as service

    model_path = Path(service.E2_MODEL_PATH)
    if not model_path.is_dir():
        pytest.skip("Original E2 checkpoint is not installed in this checkout.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).eval()
    expected = {0: "attempt", 1: "behavior", 2: "ideation", 3: "indicator", 4: "supportive", 5: "uninformative"}
    assert model.config.id2label == expected
    inputs = tokenizer(["A short smoke-test sentence."], truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)
    assert probabilities.shape == (1, 6)
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)




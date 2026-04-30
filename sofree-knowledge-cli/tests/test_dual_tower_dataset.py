import json

from sofree_knowledge.assistant.dual_tower_dataset import build_weak_supervision_samples
from sofree_knowledge.assistant.training import load_dual_tower_samples, train_dual_tower_baseline


def test_build_weak_supervision_samples_emits_positive_and_negatives():
    samples = build_weak_supervision_samples(
        documents=[
            {"doc_id": "d1", "title": "Release Flow", "summary": "Rollback checklist", "business": "Release"},
            {"doc_id": "d2", "title": "Weekly Notes", "summary": "Routine sync", "business": "General"},
        ],
        access_records=[
            {"doc_id": "d1", "user_id": "ou_1", "action": "view", "count": 2},
        ],
        messages=[
            {"message_id": "m1", "content": "release risk rollback review"},
            {"message_id": "m2", "content": "release checklist update"},
        ],
        user_profile={"role": "PM", "interests": ["release", "risk"], "business_tracks": [{"name": "Release"}]},
        target_user_id="ou_1",
    )

    assert len(samples) == 1
    assert samples[0]["doc_id"] == "d1"
    assert "role: PM" in samples[0]["user_tower_text"]
    assert "title: Release Flow" in samples[0]["positive_doc_text"]
    assert samples[0]["negative_doc_texts"]


def test_train_dual_tower_baseline_outputs_model(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    sample = {
        "user_id": "ou_1",
        "doc_id": "d1",
        "label": 1,
        "strength": 2,
        "user_tower_text": "role: PM | interests: release, risk",
        "positive_doc_text": "title: Release Flow | summary: Rollback checklist | business: Release | doc_type: file",
        "negative_doc_texts": [
            "title: Weekly Notes | summary: Routine sync | business: General | doc_type: file"
        ],
    }
    samples_file.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = load_dual_tower_samples(str(samples_file))
    result = train_dual_tower_baseline(samples=loaded, output_file=str(tmp_path / "model.json"))

    assert result["model_type"] == "dual_tower_baseline_term_weight"
    assert result["sample_summary"]["sample_count"] == 1
    assert result["quality"]["evaluated_samples"] == 1
    assert (tmp_path / "model.json").exists()

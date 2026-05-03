import json

from sofree_knowledge.assistant.dual_tower_dataset import build_weak_supervision_samples
from sofree_knowledge.assistant.training import (
    load_dual_tower_samples,
    score_dual_tower_bonus,
    score_dual_tower_with_model,
    train_dual_tower_baseline,
)


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
    assert "bonus_scale" in result
    assert "top_pair_features" in result
    assert result["quality"]["evaluated_samples"] == 1
    assert (tmp_path / "model.json").exists()


def test_score_dual_tower_with_model_filters_url_noise_and_caps_bonus():
    model = {
        "token_weights": {
            "feishu": 22.0,
            "cn": 22.0,
            "docx": 22.0,
            "project": 10.0,
            "release": 8.0,
        }
    }

    score = score_dual_tower_with_model(
        "role: pm | interests: release, project",
        (
            "title: project release | "
            "summary: see https://foo.feishu.cn/docx/abc123?openChatId=oc_x&from=feishu "
            "https://foo.feishu.cn/docx/abc123?openChatId=oc_x&from=feishu"
        ),
        model,
    )

    assert score < 20


def test_score_dual_tower_bonus_only_returns_positive_increment():
    model = {
        "bonus_scale": 0.1,
        "token_weights": {
            "release": 2.0,
            "rollback": 1.5,
            "noise": -4.0,
        },
    }

    bonus = score_dual_tower_bonus(
        "title: release rollback | summary: release checklist",
        model,
    )

    assert bonus > 0

    negative_only_bonus = score_dual_tower_bonus(
        "title: noise | summary: noise",
        model,
    )
    assert negative_only_bonus == 0


def test_score_dual_tower_with_model_can_use_pair_feature_bonus():
    model = {
        "bonus_scale": 0.2,
        "token_weights": {},
        "pair_feature_weights": {
            "role::qa": 1.0,
            "signal::acceptance": 1.0,
            "role::qa__signal::acceptance": 4.0,
        },
    }

    matched = score_dual_tower_with_model(
        "role: 测试保障岗 | interests: 回归验证, 验收条件",
        "title: release validation | summary: 需要补充验收标准和回归验证 case",
        model,
    )
    unmatched = score_dual_tower_with_model(
        "role: 后端工程岗 | interests: 接口, 数据库",
        "title: release validation | summary: 需要补充验收标准和回归验证 case",
        model,
    )

    assert matched > unmatched

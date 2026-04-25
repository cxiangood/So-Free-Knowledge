from sofree_knowledge.policy import KnowledgePolicyStore


def test_policy_defaults_to_global_review(tmp_path):
    store = KnowledgePolicyStore(tmp_path)

    assert store.get_scope("oc_test")["scope"] == "global_review"


def test_policy_can_set_chat_only(tmp_path):
    store = KnowledgePolicyStore(tmp_path)

    result = store.set_scope("oc_test", "chat_only")

    assert result["scope"] == "chat_only"
    assert KnowledgePolicyStore(tmp_path).get_scope("oc_test")["scope"] == "chat_only"

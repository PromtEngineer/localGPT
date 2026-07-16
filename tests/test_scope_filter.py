from rag_system.retrieval.scope_filter import filter_by_entity_scope


def test_named_entity_scope_removes_cross_instrument_distractor() -> None:
    documents = [
        {
            "document_id": "uuid_01_authoritative.md",
            "text": "# Borealis Calibration Standard\nThe phrase is AURORA-17.",
        },
        {
            "document_id": "uuid_08_zephyr.md",
            "text": (
                "# Zephyr calibration card\nThe phrase for the Zephyr instrument is "
                "ZEPHYR-88. Never substitute this for Borealis."
            ),
        },
    ]

    filtered = filter_by_entity_scope(
        "What is the calibration phrase for Borealis?", documents
    )

    assert [item["document_id"] for item in filtered] == [
        "uuid_01_authoritative.md"
    ]


def test_neutral_document_is_not_dropped_by_entity_scope() -> None:
    documents = [
        {
            "document_id": "borealis.md",
            "text": "The phrase for the Borealis instrument is AURORA-17.",
        },
        {
            "document_id": "glossary.md",
            "text": "Calibration phrase means an exact preflight identifier.",
        },
    ]

    assert filter_by_entity_scope("What is the Borealis phrase?", documents) == documents


def test_generic_the_instrument_phrase_is_not_an_entity() -> None:
    documents = [
        {
            "document_id": "maintenance.html",
            "text": "Borealis schedule. The instrument is serviced every 42 days.",
        }
    ]

    assert filter_by_entity_scope("How often is Borealis serviced?", documents) == documents


def test_title_declares_primary_instrument() -> None:
    documents = [
        {
            "document_id": "zephyr.md",
            "text": "# Zephyr calibration card The value is ZEPHYR-88.",
        },
        {
            "document_id": "maintenance.html",
            "text": "Borealis maintenance schedule The instrument is serviced.",
        },
    ]

    assert filter_by_entity_scope("What is the Zephyr value?", documents) == [
        documents[0]
    ]


def test_generic_query_keeps_original_candidates() -> None:
    documents = [{"document_id": "one", "text": "General policy"}]

    assert filter_by_entity_scope("Summarize the policy", documents) == documents


def test_current_query_removes_explicitly_archived_candidate() -> None:
    documents = [
        {
            "document_id": "current_borealis.md",
            "text": "# Borealis standard\nStatus: CURRENT AND AUTHORITATIVE. AURORA-17.",
        },
        {
            "document_id": "archived_borealis.md",
            "text": "# Borealis memo\nStatus: ARCHIVED — DO NOT USE. POLARIS-09.",
        },
    ]

    filtered = filter_by_entity_scope(
        "What is the current Borealis calibration phrase?", documents
    )

    assert [item["document_id"] for item in filtered] == ["current_borealis.md"]


def test_current_query_keeps_candidates_without_authority_markers() -> None:
    documents = [
        {"document_id": "borealis_one.md", "text": "# Borealis note\nValue one."},
        {"document_id": "borealis_two.md", "text": "# Borealis note\nValue two."},
    ]

    assert (
        filter_by_entity_scope("What is the current Borealis value?", documents)
        == documents
    )

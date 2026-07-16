from __future__ import annotations

import json

import fitz

from rag_system.evaluation.multimodal_harness import (
    ParserResult,
    evaluate_parser_result,
    generate_fixture_corpus,
    load_manifest,
    model_matrix,
    report_failed,
)


def test_generated_corpus_has_real_text_and_pixel_only_visual_documents(tmp_path):
    fixtures = generate_fixture_corpus(tmp_path)

    with fitz.open(fixtures["born_digital"]) as document:
        assert "AURORA-17" in document[0].get_text("text")

    for fixture_name in ("scanned_notice", "visual_topology", "visual_chart"):
        with fitz.open(fixtures[fixture_name]) as document:
            assert document.page_count == 1
            assert document[0].get_text("text").strip() == ""
            assert document[0].get_images(full=True)

    registry = fixtures["component_registry"].read_text(encoding="utf-8")
    assert "KESTREL-42" in registry
    assert set(fixtures) == {
        "born_digital",
        "scanned_notice",
        "visual_topology",
        "visual_chart",
        "component_registry",
    }


def test_manifest_exercises_ocr_visual_reasoning_and_cross_document_synthesis():
    manifest = load_manifest()
    cases = {case["name"]: case for case in manifest["retrieval_cases"]}

    assert len(manifest["index_documents"]) == 5
    assert cases["scanned_document_ocr_retrieval"]["requires_visual_evidence"]
    assert cases["diagram_relationship_retrieval"]["requires_visual_evidence"]
    assert cases["chart_reasoning_retrieval"]["requires_visual_evidence"]
    synthesis = cases["cross_document_visual_text_synthesis"]
    assert len(synthesis["expected_sources"]) == 2
    assert synthesis["query_decompose"] is True


def test_parser_scoring_requires_text_spatial_data_and_screenshot():
    check = {
        "name": "visual",
        "expected_tokens": ["Intake", "Synthesis"],
        "requires_screenshot": True,
        "requires_bounding_boxes": True,
    }
    incomplete = ParserResult(
        backend="test",
        document="diagram.pdf",
        status="passed",
        latency_ms=1,
        text="Intake Synthesis",
        bounding_box_count=2,
    )
    assert evaluate_parser_result(incomplete, check)["passed"] is False

    complete = ParserResult(
        backend="test",
        document="diagram.pdf",
        status="passed",
        latency_ms=1,
        text="Intake Synthesis",
        bounding_box_count=2,
        screenshots=["page-1.png"],
    )
    assert evaluate_parser_result(complete, check)["passed"] is True


def test_model_matrix_is_full_cartesian_product():
    matrix = model_matrix(
        ["embed-a", "embed-b"],
        ["vision-a", "vision-b"],
        ["liteparse", "docling"],
    )
    assert len(matrix) == 8
    assert len({json.dumps(item, sort_keys=True) for item in matrix}) == 8


def test_strict_failure_policy_separates_optional_parsers_from_retrieval():
    parser_failure = {
        "parsers": {"checks": [{"passed": False}]},
        "retrieval": [],
    }
    assert report_failed(parser_failure) is False
    assert report_failed(parser_failure, require_parsers=True) is True

    retrieval_failure = {
        "parsers": {"checks": [{"passed": True}]},
        "retrieval": [{"passed": False}],
    }
    assert report_failed(retrieval_failure) is True

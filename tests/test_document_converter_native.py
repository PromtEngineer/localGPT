def test_plain_text_conversion_does_not_require_docling(tmp_path):
    from rag_system.ingestion.document_converter import DocumentConverter

    source = tmp_path / "knowledge.txt"
    source.write_text("The launch code is AURORA-17.", encoding="utf-8")

    converted = DocumentConverter().convert_to_markdown(str(source))

    assert converted[0][0] == "The launch code is AURORA-17."
    assert converted[0][1]["source"] == str(source)

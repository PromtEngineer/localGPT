import unittest

from localgpt_runtime import normalize_index_options


class IndexOptionContractTests(unittest.TestCase):
    def test_snake_case_options_and_explicit_false_values_are_preserved(self):
        options = normalize_index_options(
            {
                "embedding_model": "example/embedder",
                "enrich_model": "example/enricher",
                "enable_docling_chunk": False,
                "enable_latechunk": False,
                "retrieval_mode": "dense",
            }
        )

        self.assertEqual("example/embedder", options["embedding_model"])
        self.assertEqual("example/enricher", options["enrich_model"])
        self.assertFalse(options["enable_docling_chunk"])
        self.assertFalse(options["enable_latechunk"])
        self.assertEqual("dense", options["retrieval_mode"])

    def test_legacy_ui_camel_case_maps_to_canonical_options(self):
        options = normalize_index_options(
            {
                "latechunk": True,
                "doclingChunk": True,
                "chunkSize": 768,
                "chunkOverlap": 96,
                "retrievalMode": "fts",
                "windowSize": 3,
                "enableEnrich": False,
            }
        )

        self.assertTrue(options["enable_latechunk"])
        self.assertTrue(options["enable_docling_chunk"])
        self.assertEqual(768, options["chunk_size"])
        self.assertEqual(96, options["chunk_overlap"])
        self.assertEqual("lexical", options["retrieval_mode"])
        self.assertFalse(options["enable_enrich"])


if __name__ == "__main__":
    unittest.main()

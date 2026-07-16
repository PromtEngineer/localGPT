import unittest

from rag_system.retrieval.fusion import fuse_ranked_results


class RetrievalFusionTests(unittest.TestCase):
    def test_dense_weight_changes_hybrid_order(self):
        lexical = [
            {"chunk_id": "lexical", "text": "lexical winner"},
            {"chunk_id": "dense", "text": "dense runner-up"},
        ]
        dense = [
            {"chunk_id": "dense", "text": "dense winner"},
            {"chunk_id": "lexical", "text": "lexical runner-up"},
        ]

        results = fuse_ranked_results(lexical, dense, k=2, dense_weight=0.9)

        self.assertEqual("dense", results[0]["chunk_id"])
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_fusion_uses_rank_not_incomparable_raw_scores(self):
        lexical = [
            {"chunk_id": "lexical", "score": 10_000},
            {"chunk_id": "dense", "score": 1},
        ]
        dense = [
            {"chunk_id": "dense", "score": 0.01},
            {"chunk_id": "lexical", "score": 0.001},
        ]

        results = fuse_ranked_results(lexical, dense, k=2, dense_weight=0.9)

        self.assertEqual("dense", results[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()

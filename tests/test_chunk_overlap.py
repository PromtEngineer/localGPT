import unittest

from rag_system.ingestion.overlap import add_chunk_overlap


class ChunkOverlapTests(unittest.TestCase):
    def test_prefixes_each_chunk_with_tail_of_previous_chunk(self):
        chunks = ["one two three", "four five six"]

        overlapped = add_chunk_overlap(chunks, overlap_tokens=2, max_tokens=5)

        self.assertEqual(["one two three", "two three four five six"], overlapped)


if __name__ == "__main__":
    unittest.main()

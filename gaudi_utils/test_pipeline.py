import time
import unittest

from pipeline import GaudiTextGenerationPipeline


class TestGaudiTextGenPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Overrides setUpClass from unittest to create artifacts for testing"""
        self.max_new_tokens = 100
        self.pipe = GaudiTextGenerationPipeline(
            model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
        )

        # Inputs for testing
        self.short_prompt = "Once upon a time"
        self.long_prompt = "Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them"
        self.qa_prompt = """Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP. Their superior performance over smaller models has made them incredibly useful for developers building NLP enabled applications. These models can be accessed via Hugging Face's `transformers` library, via OpenAI using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer: """

    def test_graph_compilation(self):
        """Measure latency for graph compilation."""
        start_time = time.perf_counter()
        self.pipe.compile_graph()
        end_time = time.perf_counter()
        print(f"Graph compilation latency: {end_time-start_time} seconds")

    def test_short_prompt_input(self):
        """Test llm with short prompt and measure latency and throughput"""
        start_time = time.perf_counter()
        output = self.pipe([self.short_prompt])
        end_time = time.perf_counter()
        print(f"Generated Text: {repr(output[0]['generated_text'])}")
        print(f"Latency: {end_time-start_time} seconds")
        throughput = self.max_new_tokens / (end_time - start_time)
        print(f"Throughput (including tokenization): {throughput} tokens/second")

    def test_long_prompt_input(self):
        """Test llm with long prompt and measure latency and thoughput"""
        start_time = time.perf_counter()
        output = self.pipe([self.long_prompt])
        end_time = time.perf_counter()
        print(f"Generated Text: {repr(output[0]['generated_text'])}")
        print(f"Latency: {end_time-start_time} seconds")
        throughput = self.max_new_tokens / (end_time - start_time)
        print(f"Throughput (including tokenization): {throughput} tokens/second")

    def test_qa_prompt_input(self):
        """Test llm with question answering prompt and measure latency and throughput"""
        start_time = time.perf_counter()
        output = self.pipe([self.qa_prompt])
        end_time = time.perf_counter()
        print(f"Generated Text: {repr(output[0]['generated_text'])}")
        print(f"Latency: {end_time-start_time} seconds")
        throughput = self.max_new_tokens / (end_time - start_time)
        print(f"Throughput (including tokenization): {throughput} tokens/second")


if __name__ == "__main__":
    unittest.main()

"""Tests for the benchmark harness."""

import json

import pytest

from raglib.benchmark import BenchmarkHarness, QAItem, load_qa_dataset
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document

# Import techniques to ensure they are registered
from raglib.techniques.echo_technique import EchoTechnique  # noqa: F401


class TestBenchmarkHarness:
    """Test the BenchmarkHarness functionality."""

    def test_load_qa_dataset_jsonl(self, tmp_path):
        """Test loading QA dataset from JSONL format."""
        # Create test dataset
        dataset_file = tmp_path / "test.jsonl"
        test_data = [
            {
                "question": "What is Python?",
                "answer": "A programming language",
                "context": "Python is popular"
            },
            {
                "question": "What is AI?",
                "answer": "Artificial intelligence",
                "context": "AI is everywhere"
            }
        ]

        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Load dataset
        qa_items = load_qa_dataset(dataset_file)

        assert len(qa_items) == 2
        assert qa_items[0].question == "What is Python?"
        assert qa_items[0].answer == "A programming language"
        assert qa_items[1].context == "AI is everywhere"

    def test_load_qa_dataset_csv(self, tmp_path):
        """Test loading QA dataset from CSV format."""
        import csv

        # Create test dataset
        dataset_file = tmp_path / "test.csv"
        test_data = [
            {"question": "What is Python?", "answer": "A programming language"},
            {"question": "What is AI?", "answer": "Artificial intelligence"}
        ]

        with open(dataset_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(test_data)

        # Load dataset
        qa_items = load_qa_dataset(dataset_file)

        assert len(qa_items) == 2
        assert qa_items[0].question == "What is Python?"
        assert qa_items[1].answer == "Artificial intelligence"

    def test_benchmark_harness_basic(self, tmp_path):
        """Test basic benchmark harness functionality."""
        # Create test dataset
        dataset_file = tmp_path / "test.jsonl"
        test_data = [
            {"question": "What is test?", "answer": "test", "context": "This is a test"}
        ]

        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create test corpus
        corpus = [
            Document(id="1", text="This is a test document about testing"),
            Document(id="2", text="Another test document for evaluation")
        ]

        # Use a simple technique (echo_technique if available)
        available_techniques = TechniqueRegistry.list()
        if "echo_technique" not in available_techniques:
            pytest.skip("echo_technique not available")

        technique = TechniqueRegistry.get("echo_technique")

        # Run benchmark
        harness = BenchmarkHarness(quick_mode=True, verbose=False)
        result = harness.run_benchmark(
            technique=technique,
            dataset_path=dataset_file,
            corpus_docs=corpus,
            top_k=2
        )

        # Validate result structure
        assert "timestamp" in result
        assert "technique" in result
        assert result["technique"] == "echo_technique"
        assert "metrics" in result
        assert "exact_match" in result["metrics"]
        assert "f1" in result["metrics"]
        assert "overlap" in result["metrics"]
        assert "individual_results" in result
        assert len(result["individual_results"]) == 1

    def test_benchmark_harness_with_output(self, tmp_path):
        """Test benchmark harness with output file."""
        # Create test dataset
        dataset_file = tmp_path / "test.jsonl"
        test_data = [
            {"question": "What is test?", "answer": "test"}
        ]

        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create test corpus
        corpus = [Document(id="1", text="This is a test")]

        # Use echo technique
        available_techniques = TechniqueRegistry.list()
        if "echo_technique" not in available_techniques:
            pytest.skip("echo_technique not available")

        technique = TechniqueRegistry.get("echo_technique")

        # Run benchmark with output
        output_file = tmp_path / "results.json"
        harness = BenchmarkHarness(quick_mode=True, verbose=False)

        result = harness.run_benchmark(
            technique=technique,
            dataset_path=dataset_file,
            corpus_docs=corpus,
            output_path=output_file
        )

        # Check output file was created
        assert output_file.exists()

        # Validate output file content
        with open(output_file, encoding='utf-8') as f:
            saved_result = json.load(f)

        assert saved_result["technique"] == result["technique"]
        assert saved_result["metrics"] == result["metrics"]

    def test_benchmark_metrics_calculation(self):
        """Test metric calculation logic."""
        harness = BenchmarkHarness(verbose=False)

        # Create test QA item and result
        qa_item = QAItem(question="test", answer="python programming")

        retrieved_docs = [
            Document(id="1", text="Python is a programming language"),
            Document(id="2", text="Java is another language")
        ]

        # Test metrics - fix the call signature to match new harness API
        metrics = harness._compute_metrics(qa_item, retrieved_docs)

        assert "exact_match" in metrics
        assert "f1" in metrics
        assert "overlap" in metrics

        # Should have some overlap since "programming" appears in both
        assert metrics["overlap"] > 0
        assert metrics["f1"] > 0

    def test_quick_mode_limits_examples(self, tmp_path):
        """Test that quick mode limits the number of examples."""
        # Create dataset with more than 3 examples
        dataset_file = tmp_path / "test.jsonl"
        test_data = [
            {"question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(5)
        ]

        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create test corpus
        corpus = [Document(id="1", text="test")]

        # Use echo technique
        available_techniques = TechniqueRegistry.list()
        if "echo_technique" not in available_techniques:
            pytest.skip("echo_technique not available")

        technique = TechniqueRegistry.get("echo_technique")

        # Run in quick mode
        harness = BenchmarkHarness(quick_mode=True, verbose=False)
        result = harness.run_benchmark(
            technique=technique,
            dataset_path=dataset_file,
            corpus_docs=corpus
        )

        # Should only process 3 examples in quick mode
        assert result["num_questions"] == 3
        assert len(result["individual_results"]) == 3

    def test_error_handling(self, tmp_path):
        """Test error handling in benchmark harness."""
        # Check if echo_technique is available
        available_techniques = TechniqueRegistry.list()
        if "echo_technique" not in available_techniques:
            pytest.skip("echo_technique not available")
            
        # Test with non-existent dataset
        harness = BenchmarkHarness(verbose=False)
        corpus = [Document(id="1", text="test")]

        # This should raise an error
        with pytest.raises(FileNotFoundError):
            harness.run_benchmark(
                technique=TechniqueRegistry.get("echo_technique"),
                dataset_path=tmp_path / "nonexistent.jsonl",
                corpus_docs=corpus
            )

    def test_qa_item_dataclass(self):
        """Test QAItem dataclass functionality."""
        # Test with minimal data
        item1 = QAItem(question="test?", answer="yes")
        assert item1.question == "test?"
        assert item1.answer == "yes"
        assert item1.context == ""
        assert item1.metadata == {}

        # Test with full data
        item2 = QAItem(
            question="What?",
            answer="This",
            context="Context here",
            metadata={"source": "test"}
        )
        assert item2.metadata["source"] == "test"

import subprocess
import unittest


class TestTextGenPipelineDeepSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Overrides setUpClass from unittest to create artifacts for testing"""
        self.base_command = ["python", "../gaudi_spawn.py", "--use_deepspeed", "--world_size"]

    def test_world_size_two(self):
        """Test DeepSpeed with world size of 2"""
        self.command = self.base_command + ["2", "pipeline.py"]
        result = subprocess.run(self.command)

        self.assertEqual(result.returncode, 0)

    def test_world_size_four(self):
        """Test DeepSpeed with world size of 4"""
        self.command = self.base_command + ["4", "pipeline.py"]
        result = subprocess.run(self.command)

        self.assertEqual(result.returncode, 0)

    def test_world_size_eight(self):
        """Test DeepSpeed with world size of 8"""
        self.command = self.base_command + ["8", "pipeline.py"]
        result = subprocess.run(self.command)

        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

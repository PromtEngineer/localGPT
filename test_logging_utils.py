import io
import json
import os
import sys
import tempfile
import logging
import unittest

# Ensure the project root is importable when tests run from the repository root.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag_system.utils.logging_utils import (
    StructuredLogger,
    configure_logging,
    LogContext,
    JSONLogHandler,
    system_logger,
    set_log_level,
)


class TestStructuredLogging(unittest.TestCase):
    def setUp(self):
        # Reset global logging state for each test to avoid cross-test interference.
        set_log_level('DEBUG')

    def _capture_structured_log(self, logger: StructuredLogger, event_name: str, **kwargs):
        buffer = io.StringIO()
        for handler in logger.logger.handlers:
            if isinstance(handler, JSONLogHandler):
                handler.stream = buffer

        logger.info(event_name, **kwargs)
        output = buffer.getvalue().strip()
        self.assertTrue(output, "Expected a JSON log entry")
        return json.loads(output)

    def test_structured_logger_emits_json_fields(self):
        logger = StructuredLogger('localgpt.test')
        log_record = self._capture_structured_log(logger, 'test_event', foo='bar', answer=42)

        self.assertEqual(log_record['event'], 'test_event')
        self.assertEqual(log_record['logger'], 'localgpt.test')
        self.assertEqual(log_record['foo'], 'bar')
        self.assertEqual(log_record['answer'], 42)
        self.assertIn('timestamp', log_record)
        self.assertIn('level', log_record)
        self.assertEqual(log_record['level'], 'INFO')
        self.assertIsNone(log_record.get('correlation_id'))

    def test_log_context_adds_correlation_id(self):
        logger = StructuredLogger('localgpt.test_context')
        buffer = io.StringIO()
        for handler in logger.logger.handlers:
            if isinstance(handler, JSONLogHandler):
                handler.stream = buffer

        with LogContext('abc-123'):
            logger.info('context_event', foo='baz')

        output = buffer.getvalue().strip()
        log_record = json.loads(output)

        self.assertEqual(log_record['event'], 'context_event')
        self.assertEqual(log_record['correlation_id'], 'abc-123')
        self.assertEqual(log_record['foo'], 'baz')

    def test_configure_logging_writes_json_to_file(self):
        with tempfile.NamedTemporaryFile('w+', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            configure_logging(log_level='INFO', log_file=temp_path)
            system_logger.info('file_event', debug_value=123)

            # Ensure any buffered log writes are flushed.
            logging.shutdown()

            with open(temp_path, 'r', encoding='utf-8') as file_handle:
                lines = [line.strip() for line in file_handle if line.strip()]

            self.assertTrue(lines, 'Expected at least one line in the log file')
            parsed = json.loads(lines[-1])
            self.assertEqual(parsed['event'], 'file_event')
            self.assertEqual(parsed['debug_value'], 123)
            self.assertEqual(parsed['logger'], 'localgpt.system')
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2.exceptions import SecurityError

from server import inkling_server


class ChatTemplateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "chat_template.jinja"

    def test_checksum_is_checked_before_decoding(self):
        self.path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "SHA-256 checksum mismatch for chat template"):
            inkling_server.load_template(self.path)

    def test_verified_template_cannot_mutate_input(self):
        source = b"{% if mutate %}{{ values.append(2) }}{% else %}{{ values | tojson }}{% endif %}"
        self.path.write_bytes(source)
        with patch.object(inkling_server, "TEMPLATE_SHA256", hashlib.sha256(source).hexdigest()):
            template = inkling_server.load_template(self.path)
        values = [1]
        self.assertEqual(template.render(values=values), "[1]")
        with self.assertRaises(SecurityError):
            template.render(values=values, mutate=True)
        self.assertEqual(values, [1])

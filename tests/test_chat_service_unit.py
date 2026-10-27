"""Unit tests for Gemini Chat Service using standard unittest.TestCase runner."""

import unittest
from unittest.mock import patch, MagicMock
import requests

from app import create_app
from app.services.chat_service import (
    send_chat_message,
    get_request_timeout,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    send_minimal_gemini_diagnostic,
    check_gemini_model_reachability,
)


class TestChatServiceUnittest(unittest.TestCase):
    """Chat service unit test suite runnable by standard unittest discover."""

    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_successful_response(self):
        fake_api_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Take a slow, deep breath. Let's break this study session down together."}
                        ]
                    }
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_api_response

        with patch('requests.post', return_value=mock_response):
            res = send_chat_message(
                user_message="I'm feeling stressed about my exam",
                api_key="test-key",
            )
            self.assertEqual(res["status"], "success")
            self.assertIn("Take a slow, deep breath", res["reply"])

    def test_read_timeout_handling(self):
        with patch('requests.post', side_effect=requests.exceptions.ReadTimeout("Read timed out")):
            res = send_chat_message(
                user_message="Hello MindEase",
                api_key="test-key",
            )
            self.assertEqual(res["status"], "timeout")
            self.assertEqual(res.get("timeout_type"), "response")
            self.assertIn("timed out", res["reply"].lower())

    def test_connect_timeout_handling(self):
        with patch('requests.post', side_effect=requests.exceptions.ConnectTimeout("Connect timed out")):
            res = send_chat_message(
                user_message="Hello MindEase",
                api_key="test-key",
            )
            self.assertEqual(res["status"], "timeout")
            self.assertEqual(res.get("timeout_type"), "connection")
            self.assertIn("timed out", res["reply"].lower())

    def test_quota_exceeded_429(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        with patch('requests.post', return_value=mock_response):
            res = send_chat_message(
                user_message="Hello MindEase",
                api_key="test-key",
            )
            self.assertEqual(res["status"], "quota_exceeded")
            self.assertIn("volume of requests", res["reply"].lower())

    def test_upstream_server_error_500(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch('requests.post', return_value=mock_response):
            res = send_chat_message(
                user_message="Hello MindEase",
                api_key="test-key",
            )
            self.assertEqual(res["status"], "error")
            self.assertIn("technical difficulties", res["reply"].lower())

    def test_missing_api_key(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY': ''}, clear=False):
            res = send_chat_message(
                user_message="Hello MindEase",
                api_key="",
            )
            self.assertEqual(res["status"], "service_unavailable")

    def test_crisis_language_interception(self):
        with patch('requests.post') as mock_post:
            res = send_chat_message(
                user_message="I want to kill myself",
                api_key="test-key",
            )
            self.assertEqual(res["status"], "crisis_detected")
            self.assertIn("support_resources", res)
            mock_post.assert_not_called()

    def test_timeout_configuration_resolution(self):
        self.assertEqual(get_request_timeout(), (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))
        with patch.dict('os.environ', {'GEMINI_CONNECT_TIMEOUT': '7.5', 'GEMINI_READ_TIMEOUT': '27.5'}):
            self.app.config['GEMINI_CONNECT_TIMEOUT'] = 7.5
            self.app.config['GEMINI_READ_TIMEOUT'] = 27.5
            self.assertEqual(get_request_timeout(), (7.5, 27.5))


if __name__ == '__main__':
    unittest.main()

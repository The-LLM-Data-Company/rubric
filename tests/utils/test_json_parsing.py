"""Tests for JSON parsing utilities."""

import pytest

from rubric.utils import parse_json_to_dict


class TestParseJsonToDict:
    """Tests for parse_json_to_dict function."""

    def test_parse_simple_json(self):
        """Test parsing simple JSON object."""
        result = parse_json_to_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_fence(self):
        """Test parsing JSON wrapped in markdown code fence."""
        result = parse_json_to_dict('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_with_leading_text(self):
        """Test parsing JSON with leading garbage text."""
        result = parse_json_to_dict('Here is the result: {"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_trailing_text(self):
        """Test parsing JSON with trailing garbage text."""
        result = parse_json_to_dict('{"key": "value"} I hope this helps!')
        assert result == {"key": "value"}

    def test_parse_json_with_both_leading_and_trailing_text(self):
        """Test parsing JSON with both leading and trailing text."""
        result = parse_json_to_dict('Result: {"key": "value"} Let me know if you need more.')
        assert result == {"key": "value"}

    def test_parse_json_with_trailing_newlines_and_text(self):
        """Test parsing JSON with trailing newlines and text."""
        result = parse_json_to_dict('{"overall_score": 85}\n\nNote: I evaluated based on...')
        assert result == {"overall_score": 85}

    def test_parse_nested_json(self):
        """Test parsing nested JSON objects."""
        result = parse_json_to_dict('{"outer": {"inner": "value"}}')
        assert result == {"outer": {"inner": "value"}}

    def test_parse_nested_json_with_trailing_text(self):
        """Test parsing nested JSON with trailing text."""
        result = parse_json_to_dict('{"outer": {"inner": "value"}} extra text')
        assert result == {"outer": {"inner": "value"}}

    def test_parse_json_with_braces_in_string(self):
        """Test parsing JSON where string values contain braces."""
        result = parse_json_to_dict('{"code": "if (x) { return y; }"} trailing')
        assert result == {"code": "if (x) { return y; }"}

    def test_parse_json_with_escaped_quotes(self):
        """Test parsing JSON with escaped quotes in strings."""
        result = parse_json_to_dict('{"text": "He said \\"hello\\""} extra')
        assert result == {"text": 'He said "hello"'}

    def test_parse_json_array_values(self):
        """Test parsing JSON with array values and trailing text."""
        result = parse_json_to_dict('{"items": [1, 2, 3]} done')
        assert result == {"items": [1, 2, 3]}

    def test_parse_criterion_status_with_trailing_text(self):
        """Test real-world example: criterion evaluation with trailing text."""
        result = parse_json_to_dict(
            '{"criterion_status": "MET", "explanation": "Good work"} Let me know if you need anything else.'
        )
        assert result == {"criterion_status": "MET", "explanation": "Good work"}

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(Exception):  # json.JSONDecodeError
            parse_json_to_dict("not json at all")

    def test_parse_empty_string_raises_error(self):
        """Test that empty string raises JSONDecodeError."""
        with pytest.raises(Exception):
            parse_json_to_dict("")

    def test_parse_json_with_unclosed_brace(self):
        """Test parsing JSON with unclosed brace raises error."""
        with pytest.raises(Exception):
            parse_json_to_dict('{"key": "value"')

    def test_parse_json_case_insensitive_fence(self):
        """Test parsing JSON with case-insensitive markdown fence."""
        result = parse_json_to_dict('```JSON\n{"key": "value"}\n```')
        assert result == {"key": "value"}

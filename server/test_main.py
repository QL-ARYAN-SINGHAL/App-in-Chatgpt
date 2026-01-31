# ABOUTME: Unit tests for the Time Left MCP server.
# ABOUTME: Tests the time calculation logic with various dates and edge cases.

from datetime import datetime
from unittest.mock import patch

import pytest

from main import calculate_time_remaining, tool_meta, load_widget_html, TEMPLATE_URI


class TestCalculateTimeRemaining:
    """Tests for the calculate_time_remaining function."""

    def test_returns_all_required_keys(self):
        """Verify the response contains all expected time periods."""
        result = calculate_time_remaining()

        assert "timestamp" in result
        assert "day" in result
        assert "week" in result
        assert "month" in result
        assert "year" in result

    def test_each_period_has_required_fields(self):
        """Verify each time period has label, elapsed, remaining, and detail."""
        result = calculate_time_remaining()

        for period in ["day", "week", "month", "year"]:
            assert "label" in result[period]
            assert "elapsed" in result[period]
            assert "remaining" in result[period]
            assert "detail" in result[period]

    def test_elapsed_plus_remaining_equals_100(self):
        """Verify elapsed + remaining = 100 for each period."""
        result = calculate_time_remaining()

        for period in ["day", "week", "month", "year"]:
            total = result[period]["elapsed"] + result[period]["remaining"]
            assert total == pytest.approx(100.0, abs=0.2)  # Allow rounding error

    def test_percentages_are_in_valid_range(self):
        """Verify all percentages are between 0 and 100."""
        result = calculate_time_remaining()

        for period in ["day", "week", "month", "year"]:
            assert 0 <= result[period]["elapsed"] <= 100
            assert 0 <= result[period]["remaining"] <= 100

    @patch("main.datetime")
    def test_midnight_day_progress_is_zero(self, mock_datetime):
        """At midnight, day progress should be 0%."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 0, 0, 0)

        result = calculate_time_remaining()

        assert result["day"]["elapsed"] == 0.0
        assert result["day"]["remaining"] == 100.0

    @patch("main.datetime")
    def test_noon_day_progress_is_fifty(self, mock_datetime):
        """At noon, day progress should be 50%."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0, 0)

        result = calculate_time_remaining()

        assert result["day"]["elapsed"] == 50.0
        assert result["day"]["remaining"] == 50.0

    @patch("main.datetime")
    def test_monday_midnight_week_progress_is_zero(self, mock_datetime):
        """At Monday midnight, week progress should be ~0%."""
        # June 16, 2025 is a Monday
        mock_datetime.now.return_value = datetime(2025, 6, 16, 0, 0, 0)

        result = calculate_time_remaining()

        assert result["week"]["elapsed"] == 0.0

    @patch("main.datetime")
    def test_first_of_month_progress_is_zero(self, mock_datetime):
        """At first of month midnight, month progress should be ~0%."""
        mock_datetime.now.return_value = datetime(2025, 6, 1, 0, 0, 0)

        result = calculate_time_remaining()

        assert result["month"]["elapsed"] == 0.0

    @patch("main.datetime")
    def test_jan_1_year_progress_is_zero(self, mock_datetime):
        """At Jan 1 midnight, year progress should be ~0%."""
        mock_datetime.now.return_value = datetime(2025, 1, 1, 0, 0, 0)

        result = calculate_time_remaining()

        assert result["year"]["elapsed"] == 0.0

    @patch("main.datetime")
    def test_leap_year_calculation(self, mock_datetime):
        """Verify leap year has 366 days in calculation."""
        # 2024 is a leap year, July 1 is ~halfway through
        mock_datetime.now.return_value = datetime(2024, 7, 1, 0, 0, 0)

        result = calculate_time_remaining()

        # July 1 in leap year: 182 days elapsed out of 366
        expected_percent = (182 / 366) * 100
        assert result["year"]["elapsed"] == pytest.approx(expected_percent, abs=0.2)

    @patch("main.datetime")
    def test_non_leap_year_calculation(self, mock_datetime):
        """Verify non-leap year has 365 days in calculation."""
        # 2025 is not a leap year, July 1 is ~halfway through
        mock_datetime.now.return_value = datetime(2025, 7, 1, 0, 0, 0)

        result = calculate_time_remaining()

        # July 1 in non-leap year: 181 days elapsed out of 365
        expected_percent = (181 / 365) * 100
        assert result["year"]["elapsed"] == pytest.approx(expected_percent, abs=0.2)

    @patch("main.datetime")
    def test_february_28_days_non_leap(self, mock_datetime):
        """Verify February has 28 days in non-leap year."""
        mock_datetime.now.return_value = datetime(2025, 2, 14, 12, 0, 0)

        result = calculate_time_remaining()

        # Feb 14 noon = 13.5 days out of 28
        expected_percent = (13.5 / 28) * 100
        assert result["month"]["elapsed"] == pytest.approx(expected_percent, abs=0.2)

    @patch("main.datetime")
    def test_february_29_days_leap(self, mock_datetime):
        """Verify February has 29 days in leap year."""
        mock_datetime.now.return_value = datetime(2024, 2, 14, 12, 0, 0)

        result = calculate_time_remaining()

        # Feb 14 noon = 13.5 days out of 29
        expected_percent = (13.5 / 29) * 100
        assert result["month"]["elapsed"] == pytest.approx(expected_percent, abs=0.2)

    @patch("main.datetime")
    def test_day_label_is_today(self, mock_datetime):
        """Verify day label is 'Today'."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0, 0)

        result = calculate_time_remaining()

        assert result["day"]["label"] == "Today"

    @patch("main.datetime")
    def test_day_detail_format(self, mock_datetime):
        """Verify day detail shows weekday and date."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0, 0)

        result = calculate_time_remaining()

        assert result["day"]["detail"] == "Sunday, June 15"

    @patch("main.datetime")
    def test_week_detail_format(self, mock_datetime):
        """Verify week detail shows week number."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0, 0)

        result = calculate_time_remaining()

        assert "Week" in result["week"]["detail"]

    @patch("main.datetime")
    def test_month_detail_format(self, mock_datetime):
        """Verify month detail shows month and year."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0, 0)

        result = calculate_time_remaining()

        assert result["month"]["detail"] == "June 2025"

    @patch("main.datetime")
    def test_year_detail_format(self, mock_datetime):
        """Verify year detail shows the year."""
        mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0, 0)

        result = calculate_time_remaining()

        assert result["year"]["detail"] == "2025"

    def test_timestamp_is_iso_format(self):
        """Verify timestamp is in ISO format."""
        result = calculate_time_remaining()

        # Should be parseable as ISO format
        datetime.fromisoformat(result["timestamp"])


class TestToolMeta:
    """Tests for the tool_meta function."""

    def test_returns_output_template(self):
        """Verify tool_meta returns the output template URI."""
        result = tool_meta()

        assert "openai/outputTemplate" in result
        assert result["openai/outputTemplate"] == TEMPLATE_URI

    def test_returns_widget_accessible(self):
        """Verify tool_meta marks widget as accessible."""
        result = tool_meta()

        assert "openai/widgetAccessible" in result
        assert result["openai/widgetAccessible"] is True


class TestLoadWidgetHtml:
    """Tests for the load_widget_html function."""

    def test_returns_string(self):
        """Verify widget HTML is loaded as a string."""
        # Clear the cache to ensure fresh load
        load_widget_html.cache_clear()

        result = load_widget_html()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_html_structure(self):
        """Verify widget HTML contains expected elements."""
        load_widget_html.cache_clear()

        result = load_widget_html()

        assert "<!DOCTYPE html>" in result
        assert "<html" in result
        assert "progress" in result.lower()
        assert "window.openai" in result

    def test_caching_works(self):
        """Verify the LRU cache is working."""
        load_widget_html.cache_clear()

        # Call twice
        result1 = load_widget_html()
        result2 = load_widget_html()

        # Should be the same object (cached)
        assert result1 is result2

        # Check cache info
        cache_info = load_widget_html.cache_info()
        assert cache_info.hits >= 1

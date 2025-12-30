"""Tests for CourseSearchTool.execute() method"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_with_valid_query_returns_formatted_results(self, mock_vector_store):
        """Test that execute returns properly formatted results when data exists"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="machine learning")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains expected content
        assert "AI Fundamentals" in result
        assert "machine learning" in result.lower() or "neural networks" in result.lower()
        assert "[" in result  # Should have header brackets

    def test_execute_with_empty_results_returns_no_content_message(self, mock_vector_store_empty):
        """Test that execute returns appropriate message when no results found"""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_error_returns_error_message(self, mock_vector_store_error):
        """Test that execute returns error message when search fails"""
        tool = CourseSearchTool(mock_vector_store_error)

        result = tool.execute(query="any query")

        assert "Search error" in result or "error" in result.lower()

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test that course_name filter is passed to search"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="basics", course_name="AI Fundamentals")

        mock_vector_store.search.assert_called_once_with(
            query="basics",
            course_name="AI Fundamentals",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test that lesson_number filter is passed to search"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="basics", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="basics",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test that both filters are passed to search"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="basics", course_name="AI Course", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="basics",
            course_name="AI Course",
            lesson_number=2
        )

    def test_execute_tracks_sources(self, mock_vector_store):
        """Test that sources are tracked for UI display"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="machine learning")

        # Verify sources were tracked
        assert len(tool.last_sources) > 0
        assert "text" in tool.last_sources[0]

    def test_execute_empty_results_with_course_filter_mentions_course(self, mock_vector_store_empty):
        """Test that empty results message mentions the course filter"""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute(query="topic", course_name="Some Course")

        assert "No relevant content found" in result
        assert "Some Course" in result


class TestCourseSearchToolFormatResults:
    """Tests for CourseSearchTool._format_results() method"""

    def test_format_results_includes_course_title(self, mock_vector_store, mock_search_results_with_data):
        """Test that formatted results include course title in header"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool._format_results(mock_search_results_with_data)

        assert "AI Fundamentals" in result

    def test_format_results_includes_lesson_number(self, mock_vector_store, mock_search_results_with_data):
        """Test that formatted results include lesson number"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool._format_results(mock_search_results_with_data)

        assert "Lesson 1" in result or "Lesson 2" in result

    def test_format_results_includes_document_content(self, mock_vector_store, mock_search_results_with_data):
        """Test that formatted results include the actual document content"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool._format_results(mock_search_results_with_data)

        assert "machine learning" in result.lower() or "neural networks" in result.lower()

    def test_format_results_separates_multiple_results(self, mock_vector_store, mock_search_results_with_data):
        """Test that multiple results are separated"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool._format_results(mock_search_results_with_data)

        # Should have multiple sections (two documents)
        sections = result.split("\n\n")
        assert len(sections) >= 2


class TestToolManager:
    """Tests for ToolManager functionality"""

    def test_register_tool(self, mock_vector_store):
        """Test that tools can be registered"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test that tool definitions are returned correctly"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        assert "input_schema" in definitions[0]

    def test_execute_tool_calls_correct_tool(self, mock_vector_store):
        """Test that execute_tool calls the correct registered tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test query")

        # Verify search was called on the vector store
        mock_vector_store.search.assert_called_once()

    def test_execute_tool_unknown_tool_returns_error(self):
        """Test that unknown tool names return error message"""
        manager = ToolManager()

        result = manager.execute_tool("unknown_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources(self, mock_vector_store):
        """Test that sources are retrieved from tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources_clears_sources(self, mock_vector_store):
        """Test that reset_sources clears all tool sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search and then reset
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        sources = manager.get_last_sources()
        assert len(sources) == 0

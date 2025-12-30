"""Shared pytest fixtures for RAG chatbot tests"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults


@pytest.fixture
def mock_search_results_with_data():
    """SearchResults with sample course content"""
    return SearchResults(
        documents=[
            "This is content about machine learning basics.",
            "Here we discuss neural networks and deep learning.",
        ],
        metadata=[
            {"course_title": "AI Fundamentals", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "AI Fundamentals", "lesson_number": 2, "chunk_index": 0},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_search_results_empty():
    """Empty SearchResults"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_search_results_with_error():
    """SearchResults with error"""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Search error: connection failed"
    )


@pytest.fixture
def mock_vector_store(mock_search_results_with_data):
    """Mock VectorStore that returns controlled responses"""
    store = Mock()
    store.search.return_value = mock_search_results_with_data
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_metadata.return_value = {
        "title": "AI Fundamentals",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction"},
            {"lesson_number": 1, "lesson_title": "Machine Learning Basics"},
        ],
    }
    return store


@pytest.fixture
def mock_vector_store_empty(mock_search_results_empty):
    """Mock VectorStore that returns empty results"""
    store = Mock()
    store.search.return_value = mock_search_results_empty
    store.get_lesson_link.return_value = None
    return store


@pytest.fixture
def mock_vector_store_error(mock_search_results_with_error):
    """Mock VectorStore that returns error"""
    store = Mock()
    store.search.return_value = mock_search_results_with_error
    return store


@pytest.fixture
def mock_tool_use_response():
    """Mock Anthropic response with tool_use"""
    # Create mock content block for tool use
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "machine learning"}

    response = Mock()
    response.stop_reason = "tool_use"
    response.content = [tool_use_block]

    return response


@pytest.fixture
def mock_text_response():
    """Mock Anthropic response with text only"""
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Here is my response about the course content."

    response = Mock()
    response.stop_reason = "end_turn"
    response.content = [text_block]

    return response


@pytest.fixture
def mock_final_response():
    """Mock final Anthropic response after tool execution"""
    text_block = Mock()
    text_block.type = "text"
    text_block.text = (
        "Based on the course materials, machine learning is a subset of AI."
    )

    response = Mock()
    response.stop_reason = "end_turn"
    response.content = [text_block]

    return response


@pytest.fixture
def mock_tool_use_response_2():
    """Mock second tool_use response for sequential tool calling tests"""
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_456"
    tool_use_block.input = {"query": "deep learning"}

    response = Mock()
    response.stop_reason = "tool_use"
    response.content = [tool_use_block]

    return response


@pytest.fixture
def mock_anthropic_client(mock_tool_use_response, mock_final_response):
    """Mock Anthropic client"""
    client = Mock()
    # First call returns tool_use, second returns final response
    client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
    return client


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager"""
    manager = Mock()
    manager.execute_tool.return_value = (
        "[AI Fundamentals - Lesson 1]\nThis is content about machine learning."
    )
    manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    manager.get_last_sources.return_value = [
        {"text": "AI Fundamentals - Lesson 1", "link": "https://example.com"}
    ]
    manager.reset_sources.return_value = None
    return manager


@dataclass
class TestConfig:
    """Test configuration"""

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return TestConfig()

"""Shared pytest fixtures for RAG chatbot tests"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from vector_store import SearchResults


# ============================================================================
# API Test App and Models (avoids static file mount issues from main app)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[dict]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """
    Create a test FastAPI app with API endpoints only (no static files).

    This avoids import issues with the main app.py which mounts static files
    from a path that doesn't exist in the test environment.
    """
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        """Clear a session's conversation history"""
        mock_rag_system.session_manager.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}

    @app.get("/")
    async def root():
        """Health check endpoint for testing"""
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    rag = Mock()

    # Mock session manager
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "test-session-123"
    rag.session_manager.clear_session.return_value = None

    # Mock query method
    rag.query.return_value = (
        "This is a test response about machine learning.",
        [{"text": "AI Fundamentals - Lesson 1", "link": "https://example.com/lesson1"}]
    )

    # Mock get_course_analytics
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["AI Fundamentals", "Python Basics", "Data Science 101"]
    }

    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create test FastAPI app with mocked RAG system"""
    return create_test_app(mock_rag_system)


@pytest.fixture
def client(test_app):
    """Test client for making HTTP requests"""
    return TestClient(test_app)


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


# ============================================================================
# Integration Test Fixtures (with real data)
# ============================================================================

SAMPLE_COURSE_CONTENT = """Course Title: Introduction to Machine Learning
Course Link: https://example.com/courses/ml-intro
Course Instructor: Dr. Jane Smith

Lesson 0: What is Machine Learning?
Lesson Link: https://example.com/courses/ml-intro/lesson/0
Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. In this lesson, we explore the fundamental concepts of ML, including supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.

Lesson 1: Neural Networks Basics
Lesson Link: https://example.com/courses/ml-intro/lesson/1
Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes or neurons. Each connection has a weight that adjusts as learning proceeds. Deep learning refers to neural networks with multiple hidden layers. Common architectures include convolutional neural networks (CNNs) for image processing and recurrent neural networks (RNNs) for sequential data.

Lesson 2: Training and Optimization
Lesson Link: https://example.com/courses/ml-intro/lesson/2
Training a neural network involves forward propagation, loss calculation, and backpropagation. Gradient descent is the primary optimization algorithm used to minimize the loss function. Variants like stochastic gradient descent (SGD), Adam, and RMSprop offer different trade-offs between convergence speed and stability. Regularization techniques like dropout and L2 regularization help prevent overfitting.
"""


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests with real components"""
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = ""

    def __post_init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        if not self.ANTHROPIC_API_KEY:
            self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@pytest.fixture(scope="module")
def integration_test_dir(tmp_path_factory):
    """Create a temporary directory for integration tests"""
    import shutil
    test_dir = tmp_path_factory.mktemp("integration_test")
    yield test_dir
    # Cleanup happens automatically with tmp_path_factory


@pytest.fixture(scope="module")
def integration_test_config(integration_test_dir):
    """Integration test configuration with temporary ChromaDB path"""
    chroma_path = str(integration_test_dir / "test_chroma_db")
    config = IntegrationTestConfig(CHROMA_PATH=chroma_path)
    return config


@pytest.fixture(scope="module")
def sample_course_file(integration_test_dir):
    """Create a sample course file for testing"""
    course_file = integration_test_dir / "sample_course.txt"
    course_file.write_text(SAMPLE_COURSE_CONTENT)
    return course_file


@pytest.fixture(scope="module")
def loaded_vector_store(integration_test_config, sample_course_file):
    """VectorStore with sample course data loaded"""
    from vector_store import VectorStore
    from document_processor import DocumentProcessor

    # Initialize components
    store = VectorStore(
        integration_test_config.CHROMA_PATH,
        integration_test_config.EMBEDDING_MODEL,
        integration_test_config.MAX_RESULTS
    )

    processor = DocumentProcessor(
        integration_test_config.CHUNK_SIZE,
        integration_test_config.CHUNK_OVERLAP
    )

    # Process and load the sample course
    course, chunks = processor.process_course_document(str(sample_course_file))
    store.add_course_metadata(course)
    store.add_course_content(chunks)

    yield store

    # Cleanup: clear the data
    store.clear_all_data()

"""Tests for RAG system query handling"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from rag_system import RAGSystem


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    def test_query_passes_tools_to_ai_generator(self, test_config):
        """Test that tool definitions are passed to AI generator"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            rag.query("What is machine learning?")

            # Verify generate_response was called with tools
            call_kwargs = mock_ai.generate_response.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"] is not None
            assert len(call_kwargs["tools"]) > 0

    def test_query_passes_tool_manager(self, test_config):
        """Test that tool_manager is passed to AI generator"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            rag.query("What is machine learning?")

            # Verify tool_manager was passed
            call_kwargs = mock_ai.generate_response.call_args[1]
            assert "tool_manager" in call_kwargs
            assert call_kwargs["tool_manager"] is not None

    def test_query_returns_response_and_sources(self, test_config):
        """Test that query returns both response and sources"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response about ML"
            MockAIGenerator.return_value = mock_ai

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            # Mock the tool manager's get_last_sources
            rag.tool_manager.get_last_sources = Mock(return_value=[
                {"text": "Course 1", "link": "http://example.com"}
            ])

            response, sources = rag.query("What is ML?")

            assert response == "Test response about ML"
            assert len(sources) > 0

    def test_query_resets_sources_after_retrieval(self, test_config):
        """Test that sources are reset after being retrieved"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            # Mock the tool manager methods
            mock_reset = Mock()
            rag.tool_manager.reset_sources = mock_reset

            rag.query("Test query")

            # Verify reset_sources was called
            mock_reset.assert_called_once()

    def test_query_formats_prompt_correctly(self, test_config):
        """Test that the query is formatted into a prompt"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            rag.query("What is Python?")

            # Verify the query parameter contains the user's question
            call_kwargs = mock_ai.generate_response.call_args[1]
            assert "What is Python?" in call_kwargs["query"]


class TestRAGSystemToolRegistration:
    """Tests for tool registration in RAGSystem"""

    def test_search_tool_is_registered(self, test_config):
        """Test that CourseSearchTool is registered"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            # Check tool is registered
            assert "search_course_content" in rag.tool_manager.tools

    def test_outline_tool_is_registered(self, test_config):
        """Test that CourseOutlineTool is registered"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            # Check tool is registered
            assert "get_course_outline" in rag.tool_manager.tools

    def test_both_tools_have_definitions(self, test_config):
        """Test that both tools provide valid definitions"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            definitions = rag.tool_manager.get_tool_definitions()

            assert len(definitions) == 2

            tool_names = [d["name"] for d in definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names


class TestRAGSystemSessionHandling:
    """Tests for session handling in RAGSystem"""

    def test_query_with_session_id_gets_history(self, test_config):
        """Test that conversation history is retrieved for sessions"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager') as MockSessionManager:

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai

            mock_session = Mock()
            mock_session.get_conversation_history.return_value = "Previous conversation"
            MockSessionManager.return_value = mock_session

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            rag.query("Follow up question", session_id="session123")

            # Verify history was retrieved
            mock_session.get_conversation_history.assert_called_with("session123")

            # Verify history was passed to AI
            call_kwargs = mock_ai.generate_response.call_args[1]
            assert call_kwargs["conversation_history"] == "Previous conversation"

    def test_query_without_session_has_no_history(self, test_config):
        """Test that queries without session_id have no history"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai

            mock_vector_store = Mock()
            MockVectorStore.return_value = mock_vector_store

            rag = RAGSystem(test_config)

            rag.query("Single question")

            # Verify history is None
            call_kwargs = mock_ai.generate_response.call_args[1]
            assert call_kwargs["conversation_history"] is None

"""Integration diagnostic tests to identify real failures"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestVectorStoreReal:
    """Test VectorStore with real ChromaDB"""

    def test_vector_store_initializes(self):
        """Test that VectorStore can be initialized"""
        from config import config
        from vector_store import VectorStore

        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        assert store is not None
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_vector_store_has_courses(self):
        """Test that VectorStore has courses loaded"""
        from config import config
        from vector_store import VectorStore

        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        count = store.get_course_count()
        titles = store.get_existing_course_titles()

        print(f"Course count: {count}")
        print(f"Course titles: {titles}")

        assert count > 0, "No courses found in vector store"

    def test_vector_store_search_works(self):
        """Test that search returns results"""
        from config import config
        from vector_store import VectorStore

        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        results = store.search("machine learning")

        print(f"Search error: {results.error}")
        print(f"Documents found: {len(results.documents)}")
        print(f"Is empty: {results.is_empty()}")

        if results.documents:
            print(f"First doc preview: {results.documents[0][:100]}...")

        assert results.error is None, f"Search returned error: {results.error}"


class TestCourseSearchToolReal:
    """Test CourseSearchTool with real VectorStore"""

    def test_execute_returns_content(self):
        """Test that execute returns actual content"""
        from config import config
        from vector_store import VectorStore
        from search_tools import CourseSearchTool

        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        tool = CourseSearchTool(store)
        result = tool.execute(query="what is")

        print(f"Result length: {len(result)}")
        print(f"Result preview: {result[:200] if result else 'EMPTY'}...")

        assert result is not None
        assert len(result) > 0
        assert "No relevant content" not in result or "error" not in result.lower()


class TestAnthropicAPIReal:
    """Test Anthropic API connection"""

    def test_api_key_exists(self):
        """Test that API key is configured"""
        from config import config

        print(f"API Key present: {bool(config.ANTHROPIC_API_KEY)}")
        print(
            f"API Key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}"
        )
        print(f"Model: {config.ANTHROPIC_MODEL}")

        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
        assert len(config.ANTHROPIC_API_KEY) > 10, "API key seems too short"

    def test_ai_generator_creates(self):
        """Test that AIGenerator can be created"""
        from config import config
        from ai_generator import AIGenerator

        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        assert generator is not None
        assert generator.client is not None

    def test_simple_api_call(self):
        """Test a simple API call without tools"""
        from config import config
        from ai_generator import AIGenerator

        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        try:
            response = generator.generate_response(
                query="Say 'Hello' and nothing else.", tools=None, tool_manager=None
            )
            print(f"Response: {response}")
            assert response is not None
            assert len(response) > 0
        except Exception as e:
            pytest.fail(f"API call failed: {type(e).__name__}: {e}")


class TestRAGSystemReal:
    """Test full RAG system"""

    def test_rag_system_initializes(self):
        """Test that RAGSystem can be initialized"""
        from config import config
        from rag_system import RAGSystem

        try:
            rag = RAGSystem(config)
            assert rag is not None
        except Exception as e:
            pytest.fail(f"RAGSystem init failed: {type(e).__name__}: {e}")

    def test_rag_system_query(self):
        """Test a real query through the RAG system"""
        from config import config
        from rag_system import RAGSystem

        rag = RAGSystem(config)

        try:
            response, sources = rag.query("What topics are covered in the courses?")
            print(f"Response: {response[:200] if response else 'EMPTY'}...")
            print(f"Sources: {sources}")

            assert response is not None
            assert len(response) > 0
        except Exception as e:
            pytest.fail(f"RAG query failed: {type(e).__name__}: {e}")

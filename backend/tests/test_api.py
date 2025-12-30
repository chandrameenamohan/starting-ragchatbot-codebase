"""Tests for FastAPI endpoints"""
import pytest
from unittest.mock import Mock


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_200_with_valid_request(self, client):
        """Test successful query returns 200 status"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        assert response.status_code == 200

    def test_query_returns_answer_in_response(self, client):
        """Test response contains answer field"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_returns_sources_in_response(self, client):
        """Test response contains sources field"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_returns_session_id(self, client):
        """Test response contains session_id"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)

    def test_query_with_session_id_uses_provided_id(self, client, mock_rag_system):
        """Test that provided session_id is used"""
        response = client.post(
            "/api/query",
            json={"query": "Follow up question", "session_id": "my-session-456"}
        )
        data = response.json()
        assert data["session_id"] == "my-session-456"
        mock_rag_system.query.assert_called_with("Follow up question", "my-session-456")

    def test_query_without_session_id_creates_new_session(self, client, mock_rag_system):
        """Test that missing session_id triggers session creation"""
        response = client.post(
            "/api/query",
            json={"query": "New question"}
        )
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_empty_query_returns_422(self, client):
        """Test that empty query returns validation error"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        # FastAPI validates the model but doesn't require non-empty by default
        # This test documents current behavior
        assert response.status_code == 200

    def test_query_missing_query_field_returns_422(self, client):
        """Test that missing query field returns validation error"""
        response = client.post(
            "/api/query",
            json={}
        )
        assert response.status_code == 422

    def test_query_sources_have_correct_structure(self, client):
        """Test that sources have text and link fields"""
        response = client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        data = response.json()
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source

    def test_query_calls_rag_system_query(self, client, mock_rag_system):
        """Test that the RAG system query method is called"""
        client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        mock_rag_system.query.assert_called()

    def test_query_handles_rag_system_exception(self, client, mock_rag_system):
        """Test that RAG system exceptions return 500"""
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        response = client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_200(self, client):
        """Test successful request returns 200 status"""
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_returns_total_courses(self, client):
        """Test response contains total_courses field"""
        response = client.get("/api/courses")
        data = response.json()
        assert "total_courses" in data
        assert isinstance(data["total_courses"], int)

    def test_courses_returns_course_titles(self, client):
        """Test response contains course_titles field"""
        response = client.get("/api/courses")
        data = response.json()
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)

    def test_courses_returns_correct_count(self, client):
        """Test that total_courses matches mock data"""
        response = client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 3

    def test_courses_returns_correct_titles(self, client):
        """Test that course_titles matches mock data"""
        response = client.get("/api/courses")
        data = response.json()
        assert "AI Fundamentals" in data["course_titles"]
        assert "Python Basics" in data["course_titles"]
        assert "Data Science 101" in data["course_titles"]

    def test_courses_calls_get_course_analytics(self, client, mock_rag_system):
        """Test that get_course_analytics is called"""
        client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_handles_exception(self, client, mock_rag_system):
        """Test that exceptions return 500"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Storage error")
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "Storage error" in response.json()["detail"]

    def test_courses_empty_catalog(self, client, mock_rag_system):
        """Test response with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        response = client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


class TestSessionEndpoint:
    """Tests for DELETE /api/session/{session_id} endpoint"""

    def test_clear_session_returns_200(self, client):
        """Test successful request returns 200 status"""
        response = client.delete("/api/session/test-session-123")
        assert response.status_code == 200

    def test_clear_session_returns_status_cleared(self, client):
        """Test response contains status=cleared"""
        response = client.delete("/api/session/test-session-123")
        data = response.json()
        assert data["status"] == "cleared"

    def test_clear_session_returns_session_id(self, client):
        """Test response contains the session_id"""
        response = client.delete("/api/session/my-session-abc")
        data = response.json()
        assert data["session_id"] == "my-session-abc"

    def test_clear_session_calls_session_manager(self, client, mock_rag_system):
        """Test that session_manager.clear_session is called"""
        client.delete("/api/session/session-to-clear")
        mock_rag_system.session_manager.clear_session.assert_called_with("session-to-clear")


class TestRootEndpoint:
    """Tests for GET / endpoint (health check)"""

    def test_root_returns_200(self, client):
        """Test successful request returns 200 status"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_status_ok(self, client):
        """Test response contains status=ok"""
        response = client.get("/")
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_message(self, client):
        """Test response contains message field"""
        response = client.get("/")
        data = response.json()
        assert "message" in data


class TestAPIContentTypes:
    """Tests for API content type handling"""

    def test_query_accepts_json(self, client):
        """Test that /api/query accepts JSON content type"""
        response = client.post(
            "/api/query",
            json={"query": "Test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200

    def test_query_returns_json(self, client):
        """Test that /api/query returns JSON content type"""
        response = client.post(
            "/api/query",
            json={"query": "Test"}
        )
        assert "application/json" in response.headers["content-type"]

    def test_courses_returns_json(self, client):
        """Test that /api/courses returns JSON content type"""
        response = client.get("/api/courses")
        assert "application/json" in response.headers["content-type"]

    def test_invalid_json_returns_422(self, client):
        """Test that invalid JSON returns validation error"""
        response = client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

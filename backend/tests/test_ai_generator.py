"""Tests for AIGenerator tool calling functionality"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator


class TestAIGeneratorToolCalling:
    """Tests for AIGenerator tool calling flow"""

    def test_generate_response_without_tools_returns_text(self, mock_text_response):
        """Test that generate_response returns text when no tools are used"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_text_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            result = generator.generate_response(
                query="What is Python?",
                tools=None,
                tool_manager=None
            )

            assert result == "Here is my response about the course content."

    def test_generate_response_with_tool_use_calls_tool_manager(
        self, mock_anthropic_client, mock_tool_manager
    ):
        """Test that tool_manager.execute_tool is called when Claude uses a tool"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_anthropic_client

            result = generator.generate_response(
                query="What is machine learning?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="machine learning"
            )

    def test_generate_response_returns_final_response_after_tool_use(
        self, mock_anthropic_client, mock_tool_manager
    ):
        """Test that the final response text is returned after tool execution"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_anthropic_client

            result = generator.generate_response(
                query="What is machine learning?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )

            # Should return the final response from the mock
            assert "machine learning" in result.lower() or "subset of AI" in result

    def test_generate_response_includes_tools_in_api_call(self, mock_text_response):
        """Test that tools are included in the API call parameters"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_text_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "test_tool", "description": "A test tool"}]
            generator.generate_response(
                query="Test query",
                tools=tools,
                tool_manager=None
            )

            # Verify tools were passed to the API
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"] == tools

    def test_generate_response_sets_tool_choice_auto(self, mock_text_response):
        """Test that tool_choice is set to auto when tools are provided"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_text_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "test_tool"}]
            generator.generate_response(
                query="Test query",
                tools=tools,
                tool_manager=None
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["tool_choice"] == {"type": "auto"}


class TestAIGeneratorHandleToolLoop:
    """Tests for AIGenerator._handle_tool_loop() method and sequential tool calling"""

    def test_handle_tool_loop_sends_tool_result_to_claude(
        self, mock_tool_use_response, mock_final_response, mock_tool_manager
    ):
        """Test that tool results are sent back to Claude"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            result = generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            # Verify API call was made with tool results
            call_args = mock_client.messages.create.call_args[1]
            api_messages = call_args["messages"]

            # Should have: user message, assistant tool_use, user tool_result
            assert len(api_messages) == 3
            assert api_messages[2]["role"] == "user"
            # Tool results should be in the content
            assert isinstance(api_messages[2]["content"], list)
            assert api_messages[2]["content"][0]["type"] == "tool_result"

    def test_handle_tool_loop_passes_correct_tool_id(
        self, mock_tool_use_response, mock_final_response, mock_tool_manager
    ):
        """Test that the correct tool_use_id is passed in tool_result"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            call_args = mock_client.messages.create.call_args[1]
            tool_result = call_args["messages"][2]["content"][0]

            assert tool_result["tool_use_id"] == "tool_123"

    def test_handle_tool_loop_returns_text_response(
        self, mock_tool_use_response, mock_final_response, mock_tool_manager
    ):
        """Test that the final text response is returned"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            result = generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            assert "machine learning" in result.lower() or "subset of AI" in result

    def test_two_sequential_tool_calls_makes_three_api_calls(
        self, mock_tool_use_response, mock_tool_use_response_2, mock_final_response, mock_tool_manager
    ):
        """Test that two sequential tool calls result in three API calls"""
        mock_client = Mock()
        # First round: tool_use, Second round: tool_use, Final: text
        mock_client.messages.create.side_effect = [
            mock_tool_use_response_2,  # After first tool execution
            mock_final_response         # After second tool execution (no tools)
        ]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            result = generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            # Should have made 2 API calls within the loop
            assert mock_client.messages.create.call_count == 2
            # Tool manager should have been called twice
            assert mock_tool_manager.execute_tool.call_count == 2

    def test_stops_after_max_rounds(
        self, mock_tool_use_response, mock_tool_use_response_2, mock_tool_manager
    ):
        """Test that loop terminates after MAX_TOOL_ROUNDS even if Claude wants more tools"""
        # Create a third tool_use response (should not be processed)
        tool_use_block_3 = Mock()
        tool_use_block_3.type = "tool_use"
        tool_use_block_3.name = "search_course_content"
        tool_use_block_3.id = "tool_789"
        tool_use_block_3.input = {"query": "neural networks"}

        mock_tool_use_response_3 = Mock()
        mock_tool_use_response_3.stop_reason = "tool_use"
        mock_tool_use_response_3.content = [tool_use_block_3]

        mock_client = Mock()
        # Both rounds return tool_use - loop should stop at max rounds
        mock_client.messages.create.side_effect = [
            mock_tool_use_response_2,    # Round 1 result
            mock_tool_use_response_3     # Round 2 result (should be final, no more iterations)
        ]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            # This should not raise and should terminate
            result = generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            # Should have made exactly 2 API calls (MAX_TOOL_ROUNDS)
            assert mock_client.messages.create.call_count == 2
            # Result will be empty string since the final response was tool_use
            assert result == ""

    def test_stops_on_tool_error(
        self, mock_tool_use_response, mock_final_response
    ):
        """Test that loop terminates early on tool execution error"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        # Create a tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            result = generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            # Should have made only 1 API call (stopped after error)
            assert mock_client.messages.create.call_count == 1
            # Tool manager was called once
            assert mock_tool_manager.execute_tool.call_count == 1

    def test_single_round_still_works(
        self, mock_tool_use_response, mock_final_response, mock_tool_manager
    ):
        """Test backward compatibility - single tool call round works correctly"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            result = generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            # Should have made exactly 1 API call
            assert mock_client.messages.create.call_count == 1
            # Tool manager called once
            mock_tool_manager.execute_tool.assert_called_once()
            # Should return final response text
            assert "machine learning" in result.lower() or "subset of AI" in result

    def test_first_round_includes_tools_in_params(
        self, mock_tool_use_response, mock_final_response, mock_tool_manager
    ):
        """Test that first round API call includes tools"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            tools = [{"name": "search_course_content", "description": "Search"}]
            messages = [{"role": "user", "content": "test"}]

            generator._handle_tool_loop(
                mock_tool_use_response,
                messages,
                "test system prompt",
                tools,
                mock_tool_manager
            )

            # First call should include tools (since round 1 < MAX_TOOL_ROUNDS)
            call_args = mock_client.messages.create.call_args[1]
            assert "tools" in call_args
            assert call_args["tool_choice"] == {"type": "auto"}


class TestAIGeneratorSystemPrompt:
    """Tests for AIGenerator system prompt"""

    def test_system_prompt_includes_search_tool_guidance(self):
        """Test that system prompt mentions search_course_content tool"""
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_includes_outline_tool_guidance(self):
        """Test that system prompt mentions get_course_outline tool"""
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT

    def test_conversation_history_appended_to_system(self, mock_text_response):
        """Test that conversation history is added to system prompt"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_text_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator("test-key", "test-model")
            generator.client = mock_client

            generator.generate_response(
                query="Test",
                conversation_history="Previous: Hello\nAssistant: Hi there",
                tools=None,
                tool_manager=None
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            system_content = call_kwargs["system"]

            assert "Previous conversation" in system_content
            assert "Hello" in system_content

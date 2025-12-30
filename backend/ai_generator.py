import anthropic
from typing import List, Optional, Dict

MAX_TOOL_ROUNDS = 2  # Maximum sequential tool call rounds per query


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course content for specific topics or information
2. **get_course_outline**: Get course structure including title, link, and complete lesson list

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - Course structure or outline
  - What lessons are in a course
  - Course links or lesson lists
  - Overview of course content
- Use **search_course_content** for questions about:
  - Specific course content or topics
  - Detailed information within lessons
- **Maximum 2 tool call rounds per query** - Use a second round only if first results are insufficient
- Each search should serve a distinct purpose (e.g., different courses or refining a query)
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results" or "based on the outline"

For outline queries, always include:
- Course title
- Course link
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager and tools:
            return self._handle_tool_loop(
                response=response,
                messages=api_params["messages"],
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager,
            )

        # Return direct response
        return self._extract_text_response(response)

    def _handle_tool_loop(
        self,
        response,
        messages: List[Dict],
        system_content: str,
        tools: List,
        tool_manager,
    ) -> str:
        """
        Handle iterative tool execution (max 2 rounds).

        Args:
            response: Initial response containing tool use requests
            messages: Current message history
            system_content: System prompt content
            tools: Available tool definitions
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution rounds
        """
        messages = messages.copy()
        round_count = 0

        while response.stop_reason == "tool_use" and round_count < MAX_TOOL_ROUNDS:
            round_count += 1

            # Add assistant's tool use response
            messages.append({"role": "assistant", "content": response.content})

            # Execute tools and collect results
            tool_results, has_error = self._execute_tools(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})

            # Include tools only if no error and more rounds allowed
            include_tools = not has_error and round_count < MAX_TOOL_ROUNDS
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if include_tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**next_params)

            if has_error:
                break

        return self._extract_text_response(response)

    def _execute_tools(self, response, tool_manager) -> tuple:
        """
        Execute all tool calls, return (results, has_error).

        Args:
            response: Response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, has_error boolean)
        """
        tool_results = []
        has_error = False

        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                except Exception as e:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True,
                        }
                    )
                    has_error = True

        return tool_results, has_error

    def _extract_text_response(self, response) -> str:
        """
        Extract text from response content blocks.

        Args:
            response: Claude API response

        Returns:
            Text content string
        """
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

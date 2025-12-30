#!/bin/bash

# Development script for code quality checks
# Usage: ./dev.sh [command]
#   format  - Format code with black
#   check   - Check formatting without changes
#   test    - Run tests
#   all     - Run all quality checks

set -e

COMMAND=${1:-all}

format() {
    echo "Formatting code with black..."
    uv run black backend/
    echo "Done!"
}

check() {
    echo "Checking code formatting..."
    uv run black --check backend/
    echo "All files properly formatted!"
}

run_tests() {
    echo "Running tests..."
    uv run pytest backend/tests/ -v
}

case $COMMAND in
    format)
        format
        ;;
    check)
        check
        ;;
    test)
        run_tests
        ;;
    all)
        check
        run_tests
        echo "All quality checks passed!"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: ./dev.sh [format|check|test|all]"
        exit 1
        ;;
esac

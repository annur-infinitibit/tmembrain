#!/bin/bash
# Script to run Membrain Python tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Membrain Python Test Runner${NC}"
echo "================================"

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in project root${NC}"
    echo "Some tests requiring OpenAI API may be skipped"
fi

# Check if MEMBRAIN_LIB_PATH is set
if [ -z "$MEMBRAIN_LIB_PATH" ]; then
    # Try to auto-detect
    LIB_PATH="../../target/release/libmembrain_ffi.so"
    if [ -f "$LIB_PATH" ]; then
        export MEMBRAIN_LIB_PATH="$(cd "$(dirname "$LIB_PATH")" && pwd)/$(basename "$LIB_PATH")"
        echo -e "${GREEN}Auto-detected library: $MEMBRAIN_LIB_PATH${NC}"
    else
        echo -e "${RED}Error: MEMBRAIN_LIB_PATH not set and library not found${NC}"
        echo "Please build the library first: cargo build --release"
        echo "Or set MEMBRAIN_LIB_PATH environment variable"
        exit 1
    fi
fi

# Check if library exists
if [ ! -f "$MEMBRAIN_LIB_PATH" ]; then
    echo -e "${RED}Error: Library not found at $MEMBRAIN_LIB_PATH${NC}"
    exit 1
fi

echo "Using library: $MEMBRAIN_LIB_PATH"
echo ""

# Parse arguments
TEST_TARGET="${1:-all}"

case "$TEST_TARGET" in
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        pytest -v
        ;;
    basic)
        echo -e "${GREEN}Running basic usage tests...${NC}"
        pytest test_basic_usage.py -v
        ;;
    graph)
        echo -e "${GREEN}Running graph memory tests...${NC}"
        pytest test_graph_memory.py -v
        ;;
    multiagent)
        echo -e "${GREEN}Running multi-agent tests...${NC}"
        pytest test_multi_agent.py -v
        ;;
    advanced)
        echo -e "${GREEN}Running advanced patterns tests...${NC}"
        pytest test_advanced_patterns.py -v
        ;;
    quick)
        echo -e "${GREEN}Running quick tests (no OpenAI)...${NC}"
        pytest -v -m "not openai"
        ;;
    *)
        echo -e "${RED}Unknown test target: $TEST_TARGET${NC}"
        echo "Usage: $0 [all|basic|graph|multiagent|advanced|quick]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Tests completed!${NC}"

from pathlib import Path
import sys

# Setup path - get project root dynamically
parent_root = Path(__file__).parent.parent.parent.parent  # Go up to project root from src/query_rewriter/utils/
sys.path.insert(0, str(parent_root))

from langchain_core.messages import (BaseMessage)

def invoke_llm(llm, messages:list[BaseMessage]):
    return llm.invoke(messages)


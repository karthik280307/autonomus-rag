from pathlib import Path
import sys

# Setup path - get project root dynamically
parent_root = Path(__file__).parent.parent.parent.parent  # Go up to project root from src/query_rewriter/utils/
sys.path.insert(0, str(parent_root))

from langchain_core.messages import (BaseMessage, SystemMessage, HumanMessage, AIMessage)
from src.query_rewriter.models import ChatMessage

def build_messages(history: list[ChatMessage])-> list[BaseMessage]:
    messages: list[BaseMessage]= []

    for message in history:
        
        if message.role.lower() =="user":
            messages.append( HumanMessage(content=message.content) )
        elif message.role.lower() =="system":
            messages.append( SystemMessage(content=message.content) )
        elif message.role.lower() =="assistant":
            messages.append(AIMessage(content=message.content))
    return messages


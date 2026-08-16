from dataclasses import dataclass, field

@dataclass
class ChatMessage:
    """for the history of the chat
    represent the single message of the chat"""

    role:str
    content:str

@dataclass
class QueryRewriteRequest:
    """ this will be the input the query rewriting """
    query: str
    history: list[ChatMessage]=field(default_factory=list)

@dataclass
class ReformulationResult:
    """output of the reformulation """
    original_query:str
    reformulated_query: str

@dataclass
class RewrittenQueryResult:
    """complete query rewriter"""
    original_query:str
    
    reformulated_query: str

    step_back_query: str |None =None

    expanded_queries: list[str]=field(default_factory=list)

@dataclass
class ExpansionResult:
    """ output of the expansion query logic"""

    original_query:str
    expanded_queries:list

@dataclass
class StepBackResult:
    """ output of stepback result"""
    original_query:str
    step_back_query:str
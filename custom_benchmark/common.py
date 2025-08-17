from dataclasses import dataclass

MOCK_ANSWER_PREFIX = "Mocked answer for: "

@dataclass
class CacheEvalItem:
    """
    Represents an item in the cache evaluation.
    """
    id: str
    catched: bool
    should_catch: bool



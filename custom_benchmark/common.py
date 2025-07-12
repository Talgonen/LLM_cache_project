from dataclasses import dataclass

@dataclass
class CacheEvalItem:
    """
    Represents an item in the cache evaluation.
    """
    id: str
    catched: bool
    should_catch: bool
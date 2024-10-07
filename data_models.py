from dataclasses import dataclass
from typing import Dict, Any, Callable

@dataclass
class Tool:
    definition: Dict[str, Any]
    callable: Callable
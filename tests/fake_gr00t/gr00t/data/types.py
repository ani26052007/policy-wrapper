from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionRepresentation(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class ActionType(Enum):
    NON_EEF = "non_eef"
    EEF = "eef"


class ActionFormat(Enum):
    DEFAULT = "default"


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat


@dataclass
class ModalityConfig:
    delta_indices: list[int]
    modality_keys: list[str]
    sin_cos_embedding_keys: list[str] | None = None
    mean_std_embedding_keys: list[str] | None = None
    action_configs: list[Any] | None = None

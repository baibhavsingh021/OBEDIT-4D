"""Target selection and preservation policy, independent of edit wording."""

import enum
import hashlib
from dataclasses import dataclass, field


class EditType(enum.Enum):
    APPEARANCE = "appearance"
    REPLACEMENT = "replacement"
    REMOVAL = "removal"
    BACKGROUND = "background"
    GEOMETRY = "geometry"
    STYLE = "style"


class PreservationMode(enum.Enum):
    STRICT = "strict"
    RELAXED = "relaxed"
    FREE = "free"


@dataclass
class TargetSpec:
    text_query: str = ""
    instruction: str = ""
    edit_type: EditType = EditType.APPEARANCE
    preservation: PreservationMode = PreservationMode.STRICT
    manual_masks: dict = None
    reference_images: list = None
    reference_annotation: str = ""
    protected_attributes: list = field(default_factory=lambda: [
        "shape", "structure", "pose", "articulation", "instance_identity"
    ])
    num_objects: int = 1
    is_background_only: bool = False

    def validate(self):
        if not self.text_query.strip() and not self.manual_masks:
            raise ValueError("TargetSpec requires text_query or manual_masks")
        if not self.instruction.strip():
            raise ValueError("TargetSpec requires a non-empty instruction")
        if not isinstance(self.edit_type, EditType):
            raise TypeError("edit_type must be an EditType")
        if not isinstance(self.preservation, PreservationMode):
            raise TypeError("preservation must be a PreservationMode")
        if self.num_objects < 1:
            raise ValueError("num_objects must be positive")
        if self.is_background_only and self.edit_type not in (
            EditType.BACKGROUND, EditType.REMOVAL
        ):
            raise ValueError("background-only targets require background/removal edit_type")
        if self.edit_type == EditType.GEOMETRY and self.preservation == PreservationMode.STRICT:
            raise ValueError("geometry edits cannot use strict preservation")

    def get_run_name(self):
        self.validate()
        digest = hashlib.md5(
            (self.text_query + "|" + self.instruction + "|" + self.edit_type.value).encode()
        ).hexdigest()[:8]
        return "edit_{}_{}".format(self.edit_type.value, digest)

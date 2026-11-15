"""Render the telemetry JSON Schema (C2) from the pydantic models.

The committed copy lives next to this module, in ``schema/``. The TypeScript
types are generated from that file, so it is part of the contract.
"""

import json
from pathlib import Path
from typing import Final

from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import CoreSchema

from ullm.telemetry.events import SCHEMA_VERSION, Frame

SCHEMA_PATH: Final = Path(__file__).parent / "schema" / f"telemetry.v{SCHEMA_VERSION}.json"


class _ModelTitlesOnly(GenerateJsonSchema):
    """Keep titles on models, drop them on fields.

    Field titles ("Rid", "T") add nothing to a schema read by people, and the
    TypeScript generator turns each one into a separate type alias.
    """

    def field_title_should_be_set(self, schema: CoreSchema) -> bool:
        return False


def render_schema() -> str:
    schema = Frame.model_json_schema(mode="serialization", schema_generator=_ModelTitlesOnly)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/kabir-fx/uLLM/telemetry.v{SCHEMA_VERSION}.json",
        **schema,
    }
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"

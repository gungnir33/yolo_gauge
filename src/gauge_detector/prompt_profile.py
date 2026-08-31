from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_prompts(prompts: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip())
    if not normalized or len(normalized) != len(prompts) or len(set(normalized)) != len(normalized):
        raise ValueError("prompt list must contain unique, non-empty strings")
    return normalized


@dataclass(frozen=True)
class PromptProfileMetadata:
    schema_version: int
    checkpoint_name: str
    checkpoint_sha256: str
    prompts: tuple[str, ...]
    imgsz: int

    @classmethod
    def create(
        cls,
        checkpoint: str | Path,
        prompts: list[str] | tuple[str, ...],
        imgsz: int,
    ) -> "PromptProfileMetadata":
        checkpoint_path = Path(checkpoint)
        normalized = _normalize_prompts(prompts)
        image_size = int(imgsz)
        if image_size <= 0:
            raise ValueError("imgsz must be a positive integer")
        return cls(1, checkpoint_path.name, checkpoint_sha256(checkpoint_path), normalized, image_size)


def save_profile_metadata(path: str | Path, metadata: PromptProfileMetadata) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_profile_metadata(path: str | Path) -> PromptProfileMetadata:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Prompt profile metadata not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported prompt profile schema: {payload.get('schema_version')}")
    return PromptProfileMetadata(
        schema_version=payload["schema_version"],
        checkpoint_name=str(payload["checkpoint_name"]),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        prompts=_normalize_prompts(payload["prompts"]),
        imgsz=int(payload["imgsz"]),
    )


def validate_profile(
    metadata: PromptProfileMetadata,
    checkpoint: str | Path,
    prompts: list[str] | tuple[str, ...],
    imgsz: int,
) -> None:
    checkpoint_path = Path(checkpoint)
    if metadata.checkpoint_name != checkpoint_path.name or metadata.checkpoint_sha256 != checkpoint_sha256(checkpoint_path):
        raise ValueError("checkpoint does not match prompt profile metadata")
    if metadata.prompts != _normalize_prompts(prompts):
        raise ValueError("prompt list does not match prompt profile metadata")
    if metadata.imgsz != int(imgsz):
        raise ValueError("imgsz does not match prompt profile metadata")


def prepare_prompt_profile(
    config_path: str | Path,
    output_path: str | Path,
    *,
    model_factory: Callable[..., Any] | None = None,
) -> tuple[Path, Path]:
    from .config import load_config
    from .model import YOLOEModel

    config = load_config(config_path)
    model_config = config["model"]
    prompts = config["text_prompt"]["prompts"]
    checkpoint = Path(model_config["name"])
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("Prompt embedding output must use the .npz extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    factory = model_factory or YOLOEModel
    model = factory(
        str(checkpoint),
        model_config["device"],
        model_config["imgsz"],
        model_config["half"],
    )
    model.set_text_prompts(prompts)
    saved_profile = Path(model.save_prompt_embeddings(output))
    metadata = PromptProfileMetadata.create(checkpoint, prompts, model_config["imgsz"])
    metadata_path = save_profile_metadata(output.with_suffix(".json"), metadata)
    return saved_profile, metadata_path

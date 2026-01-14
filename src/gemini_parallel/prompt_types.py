# ABOUTME: Re-export all types from google.genai.types for convenient access
# ABOUTME: Users can import from gemini_parallel.types instead of google.genai.types

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from google.genai.types import *

# Explicitly list commonly used types for better IDE support
from google.genai.types import (
    # Core content types
    Content,
    Part,
    File,
    # Generation config and related
    GenerateContentConfig,
    GenerationConfig,
    ThinkingConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    MultiSpeakerVoiceConfig,
    SpeakerVoiceConfig,
    # Response types
    GenerateContentResponse,
    Candidate,
    # Safety
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    # Schema
    Schema,
    # Modalities and media
    Modality,
    MediaResolution,
    # Tools
    Tool,
    ToolConfig,
    FunctionDeclaration,
    # Others
    HttpOptions,
    ModelSelectionConfig,
    AutomaticFunctionCallingConfig,
)


@dataclass
class PromptData:
    """
    Data structure for Gemini API prompts with multimedia support.

    Examples:
        # Text-only prompt
        prompt = PromptData(
            prompt="Explain quantum computing",
            metadata={"task_id": "task_001"}
        )

        # With audio/video positioning
        prompt = PromptData(
            prompt="Analyze this: <audio> and compare with <video>",
            audio_path=["audio1.mp3", "audio2.mp3"],
            video_path=["video1.mp4"],
            metadata={"task_id": "multimedia_task"}
        )

        # With generation config (Gemini 3)
        prompt = PromptData(
            prompt="Write a story",
            generation_config=GenerateContentConfig(
                temperature=1.0,
                maxOutputTokens=500,
                thinkingConfig=ThinkingConfig(thinking_level="high")
            ),
            metadata={"task_id": "creative_task"}
        )
    """

    # Required fields
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Audio fields
    audio_path: Optional[Union[str, List[str]]] = None
    audio_bytes: Optional[Union[bytes, List[bytes]]] = None
    audio_mime_type: Optional[Union[str, List[str]]] = None

    # Video fields
    video_url: Optional[Union[str, List[str]]] = None
    video_path: Optional[Union[str, List[str]]] = None
    video_bytes: Optional[Union[bytes, List[bytes]]] = None
    video_mime_type: Optional[Union[str, List[str]]] = None
    video_metadata: Optional[Union[Dict, List[Dict]]] = None

    # Image fields
    image_path: Optional[Union[str, List[str]]] = None
    image_bytes: Optional[Union[bytes, List[bytes]]] = None
    image_mime_type: Optional[Union[str, List[str]]] = None
    image_url: Optional[Union[str, List[str]]] = None

    # Generation config - can be dict or GenerateContentConfig
    generation_config: Optional[Union[Dict[str, Any], GenerateContentConfig]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {"prompt": self.prompt, "metadata": self.metadata}

        # Add optional fields if present
        if self.audio_path is not None:
            result["audio_path"] = self.audio_path
        if self.audio_bytes is not None:
            result["audio_bytes"] = self.audio_bytes
        if self.audio_mime_type is not None:
            result["audio_mime_type"] = self.audio_mime_type

        if self.video_url is not None:
            result["video_url"] = self.video_url
        if self.video_path is not None:
            result["video_path"] = self.video_path
        if self.video_bytes is not None:
            result["video_bytes"] = self.video_bytes
        if self.video_mime_type is not None:
            result["video_mime_type"] = self.video_mime_type
        if self.video_metadata is not None:
            result["video_metadata"] = self.video_metadata

        if self.image_path is not None:
            result["image_path"] = self.image_path
        if self.image_bytes is not None:
            result["image_bytes"] = self.image_bytes
        if self.image_mime_type is not None:
            result["image_mime_type"] = self.image_mime_type
        if self.image_url is not None:
            result["image_url"] = self.image_url

        if self.generation_config is not None:
            result["generation_config"] = self.generation_config

        return result

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Ensure metadata has task_id if not present
        if "task_id" not in self.metadata:
            import uuid

            self.metadata["task_id"] = f"task_{str(uuid.uuid4())[:8]}"

        # Set default MIME types if not specified
        if self.audio_path and not self.audio_mime_type:
            self.audio_mime_type = "audio/mp3"
        if self.video_path and not self.video_mime_type:
            self.video_mime_type = "video/mp4"
        if self.image_path and not self.image_mime_type:
            self.image_mime_type = "image/jpeg"


# Make sure __all__ is defined for proper star imports
__all__ = [
    # Custom classes
    "PromptData",
    # Re-export everything from genai.types
    "Content",
    "Part",
    "File",
    "GenerateContentConfig",
    "GenerationConfig",
    "ThinkingConfig",
    "SpeechConfig",
    "VoiceConfig",
    "PrebuiltVoiceConfig",
    "MultiSpeakerVoiceConfig",
    "SpeakerVoiceConfig",
    "GenerateContentResponse",
    "Candidate",
    "SafetySetting",
    "HarmCategory",
    "HarmBlockThreshold",
    "Schema",
    "Modality",
    "MediaResolution",
    "Tool",
    "ToolConfig",
    "FunctionDeclaration",
    "HttpOptions",
    "ModelSelectionConfig",
    "AutomaticFunctionCallingConfig",
]

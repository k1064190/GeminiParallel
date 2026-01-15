# gemini_image_generator.py
# ABOUTME: Gemini 3 Image Generation (Nano Banana) Processor
# This module provides text-to-image and image-editing capabilities using Gemini 3 models

import os
import logging
from typing import Optional, Union, List, Dict
from PIL import Image
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class GeminiImageGenerator:
    """
    Processor for Gemini 3 Image Generation (Nano Banana).

    Supports:
    - Text-to-Image generation
    - Image-to-Image editing
    - Grounded generation (Google Search)
    - High-resolution output (up to 4K)

    Models:
    - gemini-3-pro-image-preview (Nano Banana Pro): High quality, complex instructions
    - gemini-2.5-flash-image (Nano Banana): Fast, efficient
    """

    def __init__(
        self,
        key_manager,
        model_name: str = "gemini-3-pro-image-preview",
    ):
        """
        Initialize the image generator.

        Args:
            key_manager: API key manager instance
            model_name: Image generation model name
        """
        self.key_manager = key_manager
        self.model_name = model_name

        logging.info(f"GeminiImageGenerator initialized for model '{self.model_name}'")

    def _get_client(self) -> genai.Client:
        """Get an initialized client with an available key."""
        api_key = self.key_manager.get_any_available_key("image_gen")
        if not api_key:
            raise RuntimeError("No available API keys for image generation")

        # Use v1alpha for image generation features
        return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    def generate_image(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        number_of_images: int = 1,
        safety_settings: Optional[Union[Dict, List]] = None,
        use_google_search: bool = False,
    ) -> Optional[Image.Image]:
        """
        Generate an image from text.

        Args:
            prompt: Description of the image to generate
            output_file: Path to save the generated image (optional)
            aspect_ratio: "1:1", "16:9", "4:3", etc.
            image_size: "1K", "2K", "4K" (Gemini 3 Pro only)
            number_of_images: Number of images to generate (default 1)
            safety_settings: Safety configuration
            use_google_search: Enable grounding with Google Search (Gemini 3 Pro only)

        Returns:
            PIL.Image object of the first generated image, or None if failed
        """
        client = self._get_client()

        # Configure image options
        image_config = types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size if "gemini-3" in self.model_name else None,
        )

        # Configure tools
        tools = []
        if use_google_search and "gemini-3" in self.model_name:
            tools.append({"google_search": {}})

        config = types.GenerateContentConfig(
            image_config=image_config,
            response_modalities=["IMAGE"],  # Request only images
            tools=tools if tools else None,
            safety_settings=safety_settings,
            candidate_count=number_of_images,
        )

        try:
            logging.info(f"Generating image: {prompt[:50]}...")
            response = client.models.generate_content(
                model=self.model_name, contents=[prompt], config=config
            )

            generated_images = []

            # Extract images from response
            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        try:
                            img = part.as_image()
                            generated_images.append(img)
                        except Exception as e:
                            logging.error(f"Failed to process image part: {e}")

            if not generated_images:
                logging.error("No images generated in response")
                return None

            # Save images
            if output_file:
                for i, img in enumerate(generated_images):
                    if len(generated_images) > 1:
                        # Append index for multiple images
                        base, ext = os.path.splitext(output_file)
                        filename = f"{base}_{i + 1}{ext}"
                    else:
                        filename = output_file

                    img.save(filename)
                    logging.info(f"Image saved to {filename}")

            return generated_images[0]

        except Exception as e:
            logging.error(f"Image generation failed: {e}")
            return None

    def edit_image(
        self,
        prompt: str,
        input_image: Union[str, Image.Image],
        output_file: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        safety_settings: Optional[Union[Dict, List]] = None,
    ) -> Optional[Image.Image]:
        """
        Edit an existing image based on a text prompt.

        Args:
            prompt: Instructions for editing
            input_image: Path to image file or PIL Image object
            output_file: Path to save the result
            aspect_ratio: Output aspect ratio (optional)
            image_size: Output resolution (optional)

        Returns:
            PIL.Image object of the edited image
        """
        client = self._get_client()

        # Load input image
        if isinstance(input_image, str):
            try:
                img_obj = Image.open(input_image)
            except Exception as e:
                logging.error(f"Failed to load input image: {e}")
                return None
        else:
            img_obj = input_image

        # Configure output
        image_config = None
        if aspect_ratio or image_size:
            image_config = types.ImageConfig(
                aspect_ratio=aspect_ratio, image_size=image_size
            )

        config = types.GenerateContentConfig(
            image_config=image_config,
            response_modalities=["IMAGE"],
            safety_settings=safety_settings,
        )

        try:
            logging.info(f"Editing image with prompt: {prompt[:50]}...")
            response = client.models.generate_content(
                model=self.model_name, contents=[prompt, img_obj], config=config
            )

            for part in response.parts:
                if part.inline_data:
                    result_img = part.as_image()

                    if output_file:
                        result_img.save(output_file)
                        logging.info(f"Edited image saved to {output_file}")

                    return result_img

            logging.error("No image returned from edit operation")
            return None

        except Exception as e:
            logging.error(f"Image editing failed: {e}")
            return None

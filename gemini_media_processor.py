"""
Gemini Media Processor Utilities

This module provides utilities for processing and combining text prompts with audio/video files
for use with Google's Gemini API. It handles token-based positioning and various media formats.
"""

import os
import logging
import re
from google import genai
from google.genai import types


def parse_prompt_with_media_tokens(prompt: str, audio_files: list, video_files: list) -> list:
    """
    Parse prompt containing <audio> and <video> tokens and construct a content sequence.
    
    Args:
        prompt (str): Text prompt containing <audio> and <video> tokens
        audio_files: List of audio file objects/parts
        video_files: List of video file objects/parts
        
    Returns:
        list: Ordered list of content parts (text and media)
    """
    contents = []
    audio_index = 0
    video_index = 0
    
    # Find all tokens with their positions
    tokens = []
    for match in re.finditer(r'<(audio|video)>', prompt):
        tokens.append({
            'type': match.group(1),
            'start': match.start(),
            'end': match.end()
        })
    
    # Sort tokens by position
    tokens.sort(key=lambda x: x['start'])
    
    # Split text and insert media
    current_pos = 0
    
    for token in tokens:
        # Add text before token (if any)
        text_before = prompt[current_pos:token['start']].strip()
        if text_before:
            contents.append(text_before)
        
        # Add corresponding media file
        if token['type'] == 'audio' and audio_index < len(audio_files):
            contents.append(audio_files[audio_index])
            audio_index += 1
            logging.debug(f"Added audio file at position {len(contents)-1}")
        elif token['type'] == 'video' and video_index < len(video_files):
            contents.append(video_files[video_index])
            video_index += 1
            logging.debug(f"Added video file at position {len(contents)-1}")
        else:
            logging.warning(f"No {token['type']} file available for token at position {token['start']}")
        
        current_pos = token['end']
    
    # Add remaining text after last token
    remaining_text = prompt[current_pos:].strip()
    if remaining_text:
        contents.append(remaining_text)
    
    # Add any unused audio files at the end
    while audio_index < len(audio_files):
        contents.append(audio_files[audio_index])
        audio_index += 1
        logging.debug(f"Added unused audio file at end")
    
    # Add any unused video files at the end
    while video_index < len(video_files):
        contents.append(video_files[video_index])
        video_index += 1
        logging.debug(f"Added unused video file at end")
    
    return contents


def prepare_media_contents(client_instance, prompt_data: dict):
    """
    Prepare media contents from prompt data for Gemini API calls.
    
    Args:
        client_instance: An initialized genai.Client instance
        prompt_data (dict): Dictionary containing prompt and media parameters
        
    Returns:
        tuple: (contents_list, error_message)
               contents_list is None if error occurred
    """
    prompt = prompt_data.get('prompt', '')
    
    # Handle both single values and lists for all media parameters
    def ensure_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]
    
    audio_paths = ensure_list(prompt_data.get('audio_path'))
    audio_bytes_list = ensure_list(prompt_data.get('audio_bytes'))
    audio_mime_types = ensure_list(prompt_data.get('audio_mime_type', 'audio/mp3'))
    video_urls = ensure_list(prompt_data.get('video_url'))
    video_paths = ensure_list(prompt_data.get('video_path'))
    video_bytes_list = ensure_list(prompt_data.get('video_bytes'))
    video_mime_types = ensure_list(prompt_data.get('video_mime_type', 'video/mp4'))
    video_metadata_list = ensure_list(prompt_data.get('video_metadata', {}))

    # Prepare video files
    video_files = []
    
    # Process video URLs
    for i, video_url in enumerate(video_urls):
        try:
            video_metadata = video_metadata_list[i] if i < len(video_metadata_list) else {}
            if video_metadata:
                video_part = types.Part(
                    file_data=types.FileData(file_url=video_url),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
            else:
                video_part = types.Part(
                    file_data=types.FileData(file_url=video_url)
                )
            video_files.append(video_part)
            logging.debug(f"Added video URL: {video_url}")
        except Exception as e:
            error_msg = f"Failed to process video URL {video_url}: {e}"
            logging.error(error_msg)
            return None, error_msg
    
    # Process video paths
    for i, video_path in enumerate(video_paths):
        if os.path.exists(video_path):
            try:
                video_file = client_instance.files.upload(file=video_path)
                video_files.append(video_file)
                logging.debug(f"Added video file: {video_path}")
            except Exception as e:
                error_msg = f"Failed to upload video file {video_path}: {e}"
                logging.error(error_msg)
                return None, error_msg
        else:
            error_msg = f"Video file not found: {video_path}"
            logging.error(error_msg)
            return None, error_msg
    
    # Process video bytes
    for i, video_bytes in enumerate(video_bytes_list):
        try:
            video_mime_type = video_mime_types[i] if i < len(video_mime_types) else 'video/mp4'
            video_metadata = video_metadata_list[i] if i < len(video_metadata_list) else {}
            if video_metadata:
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
            else:
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type)
                )
            video_files.append(video_part)
            logging.debug(f"Added video bytes: {video_mime_type}")
        except Exception as e:
            error_msg = f"Failed to create video part from bytes: {e}"
            logging.error(error_msg)
            return None, error_msg

    # Prepare audio files
    audio_files = []
    
    # Process audio paths
    for i, audio_path in enumerate(audio_paths):
        if os.path.exists(audio_path):
            try:
                audio_file = client_instance.files.upload(file=audio_path)
                audio_files.append(audio_file)
                logging.debug(f"Added audio file: {audio_path}")
            except Exception as e:
                error_msg = f"Failed to upload audio file {audio_path}: {e}"
                logging.error(error_msg)
                return None, error_msg
        else:
            error_msg = f"Audio file not found: {audio_path}"
            logging.error(error_msg)
            return None, error_msg
    
    # Process audio bytes
    for i, audio_bytes in enumerate(audio_bytes_list):
        try:
            audio_mime_type = audio_mime_types[i] if i < len(audio_mime_types) else 'audio/mp3'
            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type=audio_mime_type
            )
            audio_files.append(audio_part)
            logging.debug(f"Added audio bytes: {audio_mime_type}")
        except Exception as e:
            error_msg = f"Failed to create audio part from bytes: {e}"
            logging.error(error_msg)
            return None, error_msg

    # Parse prompt and construct contents with proper positioning
    if prompt and ('<audio>' in prompt or '<video>' in prompt):
        # Use token-based positioning
        contents = parse_prompt_with_media_tokens(prompt, audio_files, video_files)
    else:
        # Fallback to original behavior: video + audio + text
        contents = []
        contents.extend(video_files)
        contents.extend(audio_files)
        if prompt:
            contents.append(prompt)
    
    # Ensure we have some content
    if not contents:
        error_msg = "No content provided (neither prompt nor media files)"
        logging.error(error_msg)
        return None, error_msg
    
    return contents, None 
"""
Utility functions for file operations, creating shortcuts (Windows), and saving metadata.
"""

import os
import shutil
from typing import List

import win32com.client
import json
import configparser

import torch
from PIL import Image

logger = None
from .logger import get_logger
logger = get_logger(__name__)


def load_config(config_file: str = "config.ini"):
    """
    Loads default values from an INI config file and returns them as a dictionary-like object.

    Args:
        config_file (str, optional): The path to the config file. Defaults to "config.ini".

    Returns:
        configparser.SectionProxy: A config section (dict-like) with default values.
    """
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8')
        logger.info(f"Loaded config file: {config_file}")
    else:
        logger.warning(f"Config file '{config_file}' does not exist. Using default in-code values.")
        config["DEFAULT"] = {
            "images_path": "./images",
            "labels": "people, documents, gadgets, cables, festivals, work, pets, random, nature, food, travel, architecture, art",
            "operation": "copy",
            "threshold": "0.3",
            "metadata_output": "metadata.json"
        }
    return config["DEFAULT"]

def create_directory(path):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file(src, dest):
    """
    Copies a file from src to dest, preserving metadata (timestamps, etc.).

    Args:
        src (str): Source file path.
        dest (str): Destination file path.
    """
    shutil.copy2(src, dest)

def create_shortcut(target, shortcut_path, description=None, icon_path=None):
    """
    Creates a Windows shortcut (.lnk) to the target file.

    Args:
        target (str): Absolute path to the target file.
        shortcut_path (str): Absolute path where the shortcut will be created.
        description (str, optional): Shortcut description. Defaults to None.
        icon_path (str, optional): Path to an icon file. Defaults to None.
    """
    target = os.path.abspath(target)
    shortcut_path = os.path.abspath(shortcut_path)
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(shortcut_path)
    shortcut.TargetPath = target
    shortcut.WorkingDirectory = os.path.dirname(target)
    if description:
        shortcut.Description = description
    if icon_path and os.path.exists(icon_path):
        shortcut.IconLocation = icon_path
    shortcut.save()

def perform_file_operation(src, dest_dir, operation, description=None, icon_path=None):
    """
    Performs a file operation (copy or symlink/shortcut) from src to dest_dir.

    Args:
        src (str): Source file path.
        dest_dir (str): Destination directory path.
        operation (str): Type of operation ('copy', 'symlink', etc.).
        description (str, optional): Used if creating a shortcut. Defaults to None.
        icon_path (str, optional): Used if creating a shortcut. Defaults to None.
    """
    dest = os.path.join(dest_dir, os.path.basename(src))

    if "copy" in operation:
        copy_file(src, dest)
    if "move" in operation:
        shutil.move(src, dest)
    if "symlink" in operation:
        if os.name == "nt":
            # Windows shortcut
            create_shortcut(src, dest + ".lnk", description, icon_path)
        else:
            # Unix symlink
            os.symlink(src, dest)

def save_metadata_to_json(metadata, output_file):
    """
    Saves metadata to a JSON file.

    Args:
        metadata (dict or list): Metadata to be saved.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(metadata, json_file, indent=4, ensure_ascii=False)
    logger.debug(f"Results saved to {output_file}")


def handle_images_in_group(group_folder, image_path, operation):
    """
    Moves/copies or symlinks an image into the group folder.
    """
    perform_file_operation(image_path, group_folder, operation)


def list_supported_image_files(images_path: str) -> List[str]:
    """
    Returns a list of supported image file paths in the given directory.
    Supported extensions: png, jpg, jpeg, bmp, webp

    Args:
        images_path (str): Path to a folder containing images.

    Returns:
        List[str]: List of file paths found with supported extensions.
    """
    if not os.path.isdir(images_path):
        logger.warning(f"Directory does not exist: {images_path}")
        return []
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = [
        os.path.join(images_path, f)
        for f in os.listdir(images_path)
        if f.lower().endswith(exts)
    ]
    return files


def preprocess_image(img_path: str, preprocess, device) -> torch.Tensor:
    """
    Opens and preprocesses a single image using the provided CLIP preprocess pipeline.

    Args:
        img_path (str): Path to the image file.
        preprocess (callable): The CLIP preprocess transform.
        device (str or torch.device): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: The preprocessed image on the appropriate device.
    """
    pil_img = Image.open(img_path)
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    return img_tensor

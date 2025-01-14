"""
Module that provides both synchronous and asynchronous functionality for:
- Tagging images using CLIP
- Searching for images by similarity to a text query
- Finding and grouping duplicate images by embedding similarity

All methods come in two flavors: sync (blocking) and async (using asyncio).
"""

import os
import json
nimport time

import click
import torch
import clip
import numpy as np
import asyncio
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import sys  # for async progress in one line

from .file_utils import perform_file_operation, save_metadata_to_json
from .logger import get_logger


logger = get_logger(__name__)

# PIL can raise errors on extremely large images;
# so we allow bigger than default to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = 259756800


class ImageTagger:
    """
    Class for tagging images, searching for similar images, and finding duplicates
    using the CLIP model. Provides both synchronous and asynchronous methods.
    """
    
    def __init__(self, model_name="CLIP"):
        """
        Initializes the ImageTagger instance.

        Args:
            model_name (str): Name of the model to load. Defaults to "CLIP".

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        self.model_name = model_name
        self.default_labels = [
            "people", "documents", "gadgets", "cables", "festivals", "work",
            "pets", "random", "nature", "food", "travel", "architecture", "art"
            ]
        
        if model_name == "CLIP":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully.")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    

    def _prepare_image(self, img_path):
        """
        Helper function to open and preprocess a single image (blocking).
        Used in run_in_executor for asynchronous loading as well.

        Args:
            img_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor on the correct device.
        """
        pil_img = Image.open(img_path)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return img_tensor
    
    
    def _process_image(self, image_path, labels, threshold):
        """
        Synchronous forward pass through the CLIP model for a single image
        with given text labels, returning labels above a probability threshold.

        Args:
            image_path (str): Path to the image file.
            labels (list): List of text labels.
            threshold (float): Probability threshold for tag assignment.

        Returns:
            list[dict]: A list of dictionaries with 'tag' and 'probability' keys.
        """
        try:
            image = self._prepare_image(image_path)
            text = clip.tokenize(labels).to(self.device)
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            results = [
                {"tag": label, "probability": float(prob)}
                for label, prob in zip(labels, probs)
                if prob >= threshold
                ]
            return results
        except Exception as e:
            logger.error(f"Error processing image {image_path} (sync): {str(e)}")
            return []


    def _load_images(self, images_path):
        """
        Synchronously loads and preprocesses all images from a directory.

        Args:
            images_path (str): Path to the directory containing images.

        Returns:
            tuple(list[str], torch.Tensor):
                - List of image file paths
                - A concatenated torch.Tensor of preprocessed images.
        """
        image_files = [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
        images = []
        for img_path in image_files:
            try:
                img_tensor = self._prepare_image(img_path)
                images.append(img_tensor)
            except Exception as e:
                logger.error(f"Cannot open image {img_path}: {e}")
        
        if not images:
            return [], torch.empty(0)
        
        return image_files, torch.cat(images)


    def tag_image(self, image_path, output_path=None, top_k=5, labels=None, threshold=0.3):
        """
        Synchronously generates tags for a given image using CLIP, optionally saves metadata.

        Args:
            image_path (str): Path to the image.
            output_path (str, optional): Path to save metadata. Defaults to None.
            top_k (int, optional): Number of top tags to return. Defaults to 5.
            labels (list, optional): List of possible tags. Defaults to default_labels.
            threshold (float, optional): Probability threshold. Defaults to 0.3.

        Returns:
            list[dict]: A list of assigned tags (tag + probability).
        """
        if labels is None:
            labels = self.default_labels
        
        results = self._process_image(image_path, labels, threshold)
        # Sort by probability desc
        results = sorted(results, key=lambda x: x["probability"], reverse=True)[:top_k]
        
        if output_path and results:
            metadata = {"file": image_path, "tags": results}
            save_metadata_to_json(metadata, output_path)
        
        return results
    

    def generate_tags(self, image_path, labels=None, threshold=0.3):
        """
        Synchronously generates a list of tags for a given image, returning only the tag names.

        Args:
            image_path (str): Path to the image.
            labels (list, optional): List of possible tags. Defaults to default_labels.
            threshold (float, optional): Probability threshold. Defaults to 0.3.

        Returns:
            list[str]: A list of tag names assigned to the image.
        """
        if labels is None:
            labels = self.default_labels
        
        results = self._process_image(image_path, labels, threshold)
        return [item["tag"] for item in results]
    

    def group_duplicates(self, duplicates, output_folder, operation="copy"):
        """
        Groups duplicate images into subfolders named after their most frequent tag (synchronous).

        Args:
            duplicates (list[tuple]): List of tuples (img1, img2, similarity).
            output_folder (str): Path to the folder where grouped images will be placed.
            operation (str, optional): File operation ('copy', 'move', 'symlink', etc.). Defaults to "copy".
        """
        grouped = {}
        for img1, img2, similarity in duplicates:
            group_id = None
            for existing_group_id, images_ in grouped.items():
                if img1 in images_ or img2 in images_:
                    group_id = existing_group_id
                    break
            if group_id is None:
                group_id = len(grouped) + 1
                grouped[group_id] = set()
            grouped[group_id].update([img1, img2])
        
        for group_id, images_ in grouped.items():
            tag_samples = []
            for image_path in images_:
                tags = self.generate_tags(image_path)
                tag_samples.extend(tags)
            most_common_tag = "group"
            if tag_samples:
                most_common_tag = max(set(tag_samples), key=tag_samples.count)
            
            folder_name = f"{most_common_tag}_{group_id}"
            group_folder = os.path.join(output_folder, folder_name)
            os.makedirs(group_folder, exist_ok=True)
            for image_path in images_:
                perform_file_operation(image_path, group_folder, operation)
        
        logger.info(f"Grouped duplicates into {len(grouped)} folders with meaningful tags.")
        click.echo(f"Grouped duplicates into {len(grouped)} folders with meaningful tags.")
    
  
    def search_images(self, query, images_path, top_k=5):
        """
        Synchronously searches for images most similar to a text query using CLIP embeddings.

        Args:
            query (str): The search query.
            images_path (str): Path to the directory with images.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list[tuple(str, float)]: A list of (image_path, similarity).
        """
        image_files, image_tensors = self._load_images(images_path)
        if len(image_files) == 0 or image_tensors.shape[0] == 0:
            logger.warning("No images to search in (sync).")
            return []
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors).float()
        # Tokenize query
        text_inputs = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = (text_features @ image_features.T).squeeze(0).cpu().numpy()
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(image_files[idx], float(similarities[idx])) for idx in sorted_indices]
        return results
    

    def find_duplicates(self, images_path, similarity_threshold=0.9):
        """
        Synchronously identifies duplicate images based on cosine similarity of their embeddings,
        with a progress bar indicating the comparison of pairs in one line.

        Args:
            images_path (str): Path to the folder containing images.
            similarity_threshold (float): Threshold (0-1) for considering images duplicates.

        Returns:
            list[tuple(str, str, float)]: Pairs of duplicate images and their similarity.
        """
        image_files, image_tensors = self._load_images(images_path)
        if len(image_files) == 0 or image_tensors.shape[0] == 0:
            logger.warning("No images found for duplicate checking (sync).")
            return []
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity_matrix = image_features @ image_features.T
        
        duplicates = []
        
        click.echo("Checking duplicates (sync) ...")
        num_images = len(image_files)
        total_pairs = (num_images * (num_images - 1)) // 2
        pair_gen = ((i, j) for i in range(num_images) for j in range(i + 1, num_images))
        with click.progressbar(pair_gen, label="Comparing pairs", length=total_pairs) as bar:
            for (i, j) in bar:
                similarity = float(similarity_matrix[i, j].item())
                if similarity >= similarity_threshold:
                    duplicates.append((image_files[i], image_files[j], similarity))
                    time.sleep(0.3)
        
        return duplicates
    
    
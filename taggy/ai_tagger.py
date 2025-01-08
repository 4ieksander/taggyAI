"""
Class for tagging images using CLIP and grouping duplicates.
"""
import json
import os

import click
import torch
import clip
from PIL import Image

import numpy as np

from utils.FILE_utils import perform_file_operation
from utils.logger import get_logger

logger = get_logger(__name__)

Image.MAX_IMAGE_PIXELS = 259756800

def save_metadata_to_file(output_path, metadata):
    """
    Save metadata to a JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump([metadata], json_file, indent=4)
        logger.info(f"Metadane zapisane do pliku: {output_path}")
        
class ImageTagger:
    def __init__(self, model_name="CLIP"):
        self.model_name = model_name

        if model_name == "CLIP":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def tag_image(self, image_path=None, output_path=None, top_k=5, labels=None, threshold=0.3):
        """
        Generate tags for a given image using CLIP.

        Args:
            image_path (str): Path to the image.
            output_path (str): Path to save metadata.
            top_k (int): Number of top tags to return.
            labels (list): List of possible tags.
            threshold (float): Probability threshold for tag assignment.

        Returns:
            list: Assigned tags with probabilities.
        """
        if labels is None:
            labels = [
                "people", "documents", "gadgets", "cables", "festivals", "work",
                "pets", "random", "nature", "food", "travel", "architecture", "art"
                ]
        
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text = clip.tokenize(labels).to(self.device)
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            results = [
                {"tag": label, "probability": prob}
                for label, prob in zip(labels, probs)
                if prob >= threshold
                ]
            
            results = sorted(results, key=lambda x: x["probability"], reverse=True)[:top_k]
            
            if output_path and results:
                metadata = {
                    "file": image_path,
                    "tags": results
                    }
                save_metadata_to_file(output_path, metadata)
            
            return results
        except Exception as e:
            logger.error(f"Error tagging image {image_path}: {str(e)}")
            return []
    
    def generate_tags(self, image_path, labels=None, threshold=0.3):
        """
        Generate tags for a given image using CLIP.

        Args:
            image_path (str): Path to the image.
            labels (list): List of possible tags.
            threshold (float): Probability threshold for tag assignment.

        Returns:
            list: Assigned tags.
        """
        if labels is None:
            labels = [
                "people", "documents", "gadgets", "cables", "festivals", "work",
                "pets", "random", "nature", "food", "travel", "architecture", "art"
                ]
        
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(labels).to(self.device)
        
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        assigned_labels = [label for label, prob in zip(labels, probs) if prob >= threshold]
        return assigned_labels
    
    
    def group_duplicates(self, duplicates, output_folder, operation="copy"):
        """
        Group duplicate images into folders with meaningful tags.

        Args:
            duplicates (list): List of duplicate pairs.
            output_folder (str): Path to the folder where groups will be created.
        """
        grouped = {}
        for img1, img2, similarity in duplicates:
            group_id = None
            
            # Check if either image is already in a group
            for group, images in grouped.items():
                if img1 in images or img2 in images:
                    group_id = group
                    break
            
            # If neither image is in a group, create a new group
            if group_id is None:
                group_id = len(grouped) + 1
                grouped[group_id] = set()
            
            grouped[group_id].update([img1, img2])
        
        # Create folders with meaningful tags
        for group_id, images in grouped.items():
            tag_samples = []
            for image_path in images:
                tags = self.generate_tags(image_path)
                tag_samples.extend(tags)
            
            # Determine the most common tag for the group
            most_common_tag = max(set(tag_samples), key=tag_samples.count, default=f"group_{group_id}")
            
            group_folder = os.path.join(output_folder, most_common_tag + f"_{group_id}")
            os.makedirs(group_folder, exist_ok=True)
            for image_path in images:
                perform_file_operation(image_path, group_folder, operation)
                
        logger.info(f"Grouped duplicates into {len(grouped)} folders with meaningful tags.")
        click.echo(f"Grouped duplicates into {len(grouped)} folders with meaningful tags.")
    
    def load_images(self, images_path):
        """
        Load and preprocess images from the given folder.
        """
        image_files = [
            os.path.join(images_path, f) for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        images = [self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device) for img_path in image_files]
        return image_files, torch.cat(images)
    
    
    def search_images(self, query, images_path, top_k=5):
        image_files, image_tensors = self.load_images(images_path)
        
        # Vectorize the query and images
        text_inputs = clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors).float()
            text_features = self.model.encode_text(text_inputs).float()
        
        # Normalise feature vectors
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Normalize features
        similarities = (text_features @ image_features.T).squeeze(0).cpu().numpy()
        
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(image_files[idx], similarities[idx]) for idx in sorted_indices]
        
        return results
    
    def find_duplicates(self, images_path, similarity_threshold=0.9):
        """
        Identify duplicate images based on cosine similarity.

        Args:
            images_path (str): Path to the folder containing images.
            similarity_threshold (float): Threshold for cosine similarity to consider images as duplicates.

        Returns:
            list: Pairs of duplicate images with their similarity scores.
        """
        image_files, image_tensors = self.load_images(images_path)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors).float()

        # Normalise feature vectors
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarities
        similarity_matrix = image_features @ image_features.T

        # Extract duplicate pairs
        duplicates = []
        num_images = len(image_files)
        for i in range(num_images):
            for j in range(i + 1, num_images):
                similarity = similarity_matrix[i, j].item()
                if similarity >= similarity_threshold:
                    duplicates.append((image_files[i], image_files[j], similarity))

        return duplicates
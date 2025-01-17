"""
Module that provides both synchronous functionality for:
- Tagging images using CLIP
- Searching for images by similarity to a text query
- Finding and grouping duplicate images by embedding similarity
- Proposing the best images in each group based on multiple metrics:
  - Naive "sharpness" via Laplacian
  - Face-detection-based sharpness
- Optionally placing non-duplicates in a 'non_duplicates' folder
- Saving grouping results to JSON

"""

import os
import click
import torch
import clip
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from .file_utils import (
    create_directory, perform_file_operation,
    save_metadata_to_json,
    list_supported_image_files,
    preprocess_image
    )
from .logger import get_logger


logger = get_logger(__name__)

Image.MAX_IMAGE_PIXELS = 259756800



class ImageTagger:
    """
    A class used to tag images, search for images by similarity, find and group duplicate images,
    and propose the best images in each group based on multiple metrics.

    Attributes:
        model_name (str): The name of the model to load.
        labels (list): List of possible tags.
        face_cascade_path (str): The path to the Haar cascade file for face detection.
        device (str): The device to run the model on (CPU or GPU).
        model (torch.nn.Module): The loaded CLIP model.
        preprocess (callable): The preprocessing function for the CLIP model.
    """

    def __init__(self,
                 model_name: str = "CLIP",
                 labels: list = None,
                 face_cascade_path: str = None):
        """
        Initializes the ImageTagger instance.

        Args:
            model_name (str): Name of the model to load. Defaults to "CLIP".
            labels (list, optional): List of possible tags. Defaults to None.
            face_cascade_path (str, optional): Path to the Haar cascade file for face detection. Defaults to None.
        """
        self.labels = labels
        self.model_name = model_name
        self.face_cascade_path = face_cascade_path
        
        if torch.cuda.is_available():
            logger.critical(f"CUDA available, device: {torch.cuda.get_device_name(0)}")
        else:
            logger.critical("CUDA not available, using CPU.")

        if model_name == "CLIP":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully.")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if cv2 is None:
            logger.warning("OpenCV not installed. Face-detection-based metrics may be unavailable.")
    
    
    
    # ---------------------------------------
    # Tagging images - command 'tag'
    # ---------------------------------------
    def tag_image(self,
                  image_path: str,
                  output_path: str = None,
                  top_k: int = 5,
                  labels=None,
                  threshold: float = 0.3,
                  operation: str = None,
                  output_folder: str = None):
        """
        Generates tags for a given image using CLIP, optionally saves metadata.

        Args:
            image_path (str): Path to the image.
            output_path (str, optional): Path to save metadata. Defaults to None.
            top_k (int, optional): Number of top tags to return. Defaults to 5.
            labels (list) List of possible tags.
            threshold (float, optional): Probability threshold. Defaults to 0.3.

        Returns:
            list[dict]: A list of assigned tags (tag + probability).
        """
        results = self._process_image(image_path, threshold,  labels)
        results = sorted(results, key=lambda x: x["probability"], reverse=True)[:top_k]
        
        if output_folder and results:
            assigned_labels = [r["tag"] for r in results]
            for label in assigned_labels:
                label_dir = os.path.join(output_folder, label)
                create_directory(label_dir)
                perform_file_operation(image_path, label_dir, operation)
                
        if output_path and results:
            metadata = {"file": image_path, "tags": results}
            save_metadata_to_json(metadata, output_path)
        return results
    
    
    # ---------------------------------------
    # Search  image(s) similarity to query - command 'search'
    # ---------------------------------------
    def search_images(self, query: str, images_path: str, top_k: int = 5, output_path: str = None, operation: str = None):
        """
        Searches for images similar to a text query using the CLIP model.

        Args:
            query (str): The text query to search for.
            images_path (str): Path to the directory containing images.
            top_k (int, optional): Number of top similar images to return. Defaults to 5.
            output_path (str, optional): Path to save files what was found. Defaults to None.
            operation (str, optional): File operation. Defaults to "copy".
            
        Returns:
            list[tuple[str, float]]: A list of tuples containing image file paths and their similarity scores.
        """
        image_files, image_tensors = self._load_images(images_path)
        if len(image_files) == 0 or image_tensors.shape[0] == 0:
            logger.warning("No images to search in (sync).")
            return []
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors).float()
        text_inputs = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = (text_features @ image_features.T).squeeze(0).cpu().numpy()
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(image_files[idx], float(similarities[idx])) for idx in sorted_indices]
        if output_path and results:
            for img_path, _ in results:
                perform_file_operation(img_path, output_path, operation)
            logger.info(f"Files saved to {output_path} with operation: {operation}")
        return results
    
    
    # ---------------------------------------
    # Find duplicates - command 'duplicates'
    # ---------------------------------------
    def find_duplicates(self, images_path: str, similarity_threshold: float = 0.9):
        """
        Identifies duplicate images by cosine similarity of embeddings, synchronously.

        This method loads images from the specified directory, computes their embeddings using the CLIP model,
        and then calculates the cosine similarity between each pair of images. If the similarity exceeds the
        specified threshold, the pair is considered a duplicate.

        Args:
            images_path (str): Path to the folder containing images.
            similarity_threshold (float): Threshold for considering images as duplicates, ranging from 0 to 1. Defaults to 0.9.

        Returns:
            list[tuple[str, str, float]]: A list of tuples where each tuple contains two image paths and their similarity score.
        """
        logger.info("Checking duplicates (sync) ...")
        image_files, image_tensors = self._load_images(images_path)
        if len(image_files) == 0 or image_tensors.shape[0] == 0:
            logger.warning("No images found for duplicate checking (sync).")
            return []
        
        logger.info(f"Loaded {len(image_files)} images for duplicate checking.")
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors).float()
        
        logger.info("Image features extracted.")
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity_matrix = image_features @ image_features.T
        logger.info("Similarity matrix computed.")
        
        duplicates = []
        num_images = len(image_files)
        total_pairs = (num_images * (num_images - 1)) // 2
        logger.info(f"Comparing {total_pairs} pairs of images ...")
        
        pair_gen = ((i, j) for i in range(num_images) for j in range(i + 1, num_images))
        logger.info("Starting comparison ...")
        
        with click.progressbar(pair_gen, label="Comparing pairs", length=total_pairs) as bar:
            for (i, j) in bar:
                similarity = float(similarity_matrix[i, j].item())
                if similarity >= similarity_threshold:
                    duplicates.append((image_files[i], image_files[j], similarity))
        
        return duplicates
    
    
    # ---------------------------------------
    # Group duplicates - command 'duplicates'
    # ---------------------------------------
    def group_duplicates(self,
                         duplicates: list,
                         output_folder: str,
                         operation: str = "copy",
                         propose_best: bool = True,
                         all_images: list = None,
                         best_scoring_method: str = "advanced"):
        """
        High-level method to group duplicates into subfolders, also handle non-duplicates.
        - Finds groups from duplicates
        - Processes each group (common tag, best images)
        - Places non-duplicate images in a 'non_duplicates' subfolder
        - Saves a JSON summary of everything

        Args:
            duplicates (list[tuple]): (img1, img2, similarity).
            output_folder (str): Where grouped images will be placed.
            operation (str, optional): File operation. Defaults to "copy".
            propose_best (bool, optional): Whether to measure quality. Defaults to True.
            all_images (list[str], optional): If provided, separate out non-duplicates. Defaults to None.
            best_scoring_method (str, optional): Method to score the best images. Defaults to "advanced".
        """
        grouped = _find_duplicate_groups(duplicates)
        group_records = self._process_duplicate_groups(
            grouped,
            output_folder,
            operation,
            propose_best,
            best_scoring_method=best_scoring_method
            )
        
        results_data = {
            "duplicates":     group_records,
            "non_duplicates": []
            }
        
        if all_images:
            duplicates_files = set()
            for images_ in grouped.values():
                duplicates_files.update(images_)
            non_duplicates_set = set(all_images) - duplicates_files
            
            if non_duplicates_set:
                non_dup_folder = os.path.join(output_folder, "non_duplicates")
                os.makedirs(non_dup_folder, exist_ok=True)
                
                for image_path in non_duplicates_set:
                    perform_file_operation(image_path, non_dup_folder, operation)
                    if propose_best:
                        gray = _load_image(image_path)
                        q = self._combined_image_score(gray)
                        results_data["non_duplicates"].append({
                            "path":  image_path,
                            "score": q
                            })
                    else:
                        results_data["non_duplicates"].append({"path": image_path})
                        
                logger.info(f"Non-duplicates placed in {non_dup_folder}")
                if output_folder:
                    save_metadata_to_json(results_data, os.path.join(non_dup_folder, "non_duplicates.json"))
        
    
    # ---------------------------------------
    # Helper method used by commands "tag", "search" and "duplicates"
    # ---------------------------------------
    def _process_image(self, image_path: str, threshold: float, labels: list = None):
        """
        Forward pass (synchronous) through the CLIP model for a single image,
        returning labels above a probability threshold.

        Args:
            image_path (str): Path to the image file.
            labels (list): List of text labels.
            threshold (float): Probability threshold for tag assignment.

        Returns:
            list[dict]: A list of dictionaries with 'tag' and 'probability' keys.
        """
        if not labels:
            labels = self.labels
        try:
            image_tensor = preprocess_image(image_path, self.preprocess, self.device)
            text = clip.tokenize(labels).to(self.device)
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image_tensor, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()
            
            if probs.ndim == 0:
                probs = [float(probs)]
            elif probs.ndim == 1:
                probs = probs.tolist()
            else:
                logger.error(f"Unexpected shape for probabilities: {probs.shape} in image {image_path}")
                return []
            
            results = [
                {"tag": label, "probability": float(prob)}
                for label, prob in zip(labels, probs)
                if prob >= threshold
                ]
            
            return results
        except Exception as e:
            logger.error(f"Error processing image {image_path} (sync): {str(e)}")
            return []
    
    
    
    # ---------------------------------------
    # Helper methods for commands "duplicates" and "search"
    # ---------------------------------------
    def _load_images(self, images_path: str):
        """
        Loads all images from a directory and preprocesses them into a Tensor (synchronously).

        Args:
            images_path (str): Path to the directory containing images.

        Returns:
            tuple(list[str], torch.Tensor):
                - List of image file paths
                - A concatenated torch.Tensor of preprocessed images
        """
        image_files = list_supported_image_files(images_path)
        if not image_files:
            return [], torch.empty(0)
        
        images = []
        with click.progressbar(image_files, label="Loading images") as bar:
            for img_path in bar:
                try:
                    img_tensor = preprocess_image(img_path, self.preprocess, self.device)
                    images.append(img_tensor)
                except Exception as e:
                    logger.error(f"Cannot open image {img_path}: {e}")
        
        if not images:
            return [], torch.empty(0)
        
        return image_files, torch.cat(images)

    
    
    # ---------------------------------------
    # Helper methods for command "duplicates"
    # ---------------------------------------
    def _process_duplicate_groups(self,
                                  grouped: dict,
                                  output_folder: str,
                                  operation: str = "copy",
                                  propose_best: bool = True,
                                  best_scoring_method: str = "advanced"):
        """
        Processes each group to:
        - Find the most common tag
        - Optionally measure quality to pick best images
        - Perform file operation and assemble result data

        Args:
            grouped (dict[int, set]): group_id => set of image paths
            output_folder (str): Destination folder
            operation (str, optional): File op ('copy', 'move', etc.). Defaults to "copy".
            propose_best (bool, optional): If True, measure image quality. Defaults to True.
            best_scoring_method (str, optional): "advanced" or "laplacian".

        Returns:
            list[dict]: A list of group records with info about images & best images
        """
        group_records = []
        for group_id, images_ in grouped.items():
            # gather tags
            tag_samples = []
            for image_path in images_:
                tags = self._generate_tags(image_path)
                tag_samples.extend(tags)
            most_common_tag = "group"
            if tag_samples:
                most_common_tag = max(set(tag_samples), key=tag_samples.count)
            
            # measure quality
            scored_images = []
            if propose_best:
                for image_path in images_:
                    gray_img = _load_image(image_path)
                    if gray_img is None:
                        score = 0.0
                    elif best_scoring_method == "advanced":
                        score = self._combined_image_score(gray_img)
                    else:
                        score = _calculate_sharpness(gray_img)
                    scored_images.append((image_path, score))
                scored_images.sort(key=lambda x: x[1], reverse=True)
                best_images = scored_images[:3]
            else:
                best_images = []
            
            # create group subfolder
            folder_name = f"{most_common_tag}_{group_id}"
            group_folder = os.path.join(output_folder, folder_name)
            os.makedirs(group_folder, exist_ok=True)
            
            # move/copy/symlink images
            group_images_info = []
            if propose_best:
                for (img_path, q) in scored_images:
                    perform_file_operation(img_path, group_folder, operation)
                    group_images_info.append({"path": img_path, "score": q})
            else:
                for img_path in images_:
                    perform_file_operation(img_path, group_folder, operation)
                    group_images_info.append({"path": img_path})
            if output_folder:
                logger.info(f"Group {group_id} ({len(images_)} images) => {group_folder}")
            record = {
                "group_id":    group_id,
                "tag":         most_common_tag,
                "folder_name": folder_name,
                "images":      group_images_info
                }
            if best_images:
                record["best_images"] = [
                    {"path": img_path, "score": score}
                    for (img_path, score) in best_images
                    ]
            group_records.append(record)
        
        logger.info(f"Operation on files: {operation}")
        if output_folder and group_records:
            save_metadata_to_json(group_records, os.path.join(output_folder, "grouped_images.json"))
        return group_records
    
    
    def _generate_tags(self, image_path: str, threshold: float = 0.3):
        """
        Generates a list of tag names for the given image.

        Args:
            image_path (str): Path to the image.
            threshold (float, optional): Probability threshold. Defaults to 0.3.

        Returns:
            list[str]: A list of tag names assigned to the image.
        """
        results = self._process_image(image_path, threshold)
        return [item["tag"] for item in results]

    
    def _combined_image_score(self, gray: np.ndarray):
        """
        Computes a combined image quality score based on sharpness and face detection.
        
        This method calculates the sharpness of the entire image using the variance of the Laplacian.
        If a Haar cascade file for face detection is available, it also calculates the sharpness of detected faces
        and combines these scores to produce a final quality score.
        
        Args:
            gray (np.ndarray): Grayscale image array.
        
        Returns:
            float: Combined image quality score.
        """
        base_sharpness = _calculate_sharpness(gray)
        
        if not os.path.exists(self.face_cascade_path):
            logger.warning(f"Haar cascade not found: {self.face_cascade_path}. Using sharpness only.")
            return base_sharpness * 0.3
        
        face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            return base_sharpness * 0.3
        
        face_sharpness = [_calculate_sharpness(gray[y:y + h, x:x + w]) for (x, y, w, h) in faces]
        avg_face_sharpness = sum(face_sharpness) / len(face_sharpness) if face_sharpness else 0.0
        
        return (avg_face_sharpness * 0.5) + (len(faces) * 0.2)



# ---------------------------------------
# Helper functions for command 'duplicates'
# ---------------------------------------
def _find_duplicate_groups(duplicates: list):
    """
    Builds a dictionary grouping sets of duplicates. Each group has a numeric ID.

    Args:
        duplicates (list[tuple]): e.g. list of (img1, img2, similarity).

    Returns:
        dict[int, set]: A mapping from group_id => set of image paths
    """
    grouped = {}
    with click.progressbar(duplicates, label="Grouping duplicates") as bar:
        for img1, img2, similarity in bar:
            group_id = None
            for existing_group_id, images_ in grouped.items():
                if img1 in images_ or img2 in images_:
                    group_id = existing_group_id
                    break
            if group_id is None:
                group_id = len(grouped) + 1
                grouped[group_id] = set()
            grouped[group_id].update([img1, img2])
    return grouped
    
 
def _measure_image_quality(image_path: str):
    """
    Naive 'sharpness' measure using the variance of the Laplacian
    on the entire image. Higher => sharper image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        float: Variance of Laplacian (0 if error or if cv2 missing).
    """
    if cv2 is None:
        return 0.0
    
    try:
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            return 0.0
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception as e:
        logger.error(f"Error measuring quality for {image_path}: {e}")
        return 0.0


def _load_image(image_path: str):
    """
    Loads an image from the specified path and converts it to a grayscale image.

    This function uses OpenCV to read the image data from the file, decode it, and convert it to grayscale.
    If OpenCV is not available or the image cannot be loaded, it returns None.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image array if successful, None otherwise.
    """
    if cv2 is None:
        return None
    try:
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            raise ValueError("Invalid image data.")
        return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None
    
    
def _calculate_sharpness(img_gray: np.ndarray):
    """
    Calculates the sharpness of a grayscale image using the variance of the Laplacian.

    This function computes the Laplacian of the image and returns the variance, which is a measure of sharpness.
    Higher variance indicates a sharper image.

    Args:
        img_gray (np.ndarray): Grayscale image array.

    Returns:
        float: Variance of the Laplacian, representing the image sharpness.
    """
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
# ---------------------------------------
# Aleksander Okrasa
# Please dont share this code with anyone
# ---------------------------------------
"""
Klasa ImageTagger oparta na modelu CLIP, obsługująca wczytywanie
i przetwarzanie obrazów w celu przypisywania etykiet.
"""
import json
import os
import torch
import clip
from PIL import Image
import numpy as np
import piexif
from utils.logger import get_logger

logger = get_logger(__name__)


def save_metadata_to_file(output_path, metadata):
	"""
	Zapisuje metadane w formacie JSON.
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
		try:
			if self.model_name == "CLIP":
				if not labels:
					labels = ["people", "cat", "dog", "meme", "other"]
				tags = self.predict_with_clip(image_path, labels)
				
				assigned_labels = [r["tag"] for r in tags if r["probability"] >= threshold]
				
				if output_path and assigned_labels:
					metadata = {
						"file": image_path,
						"tags": tags
						}
					save_metadata_to_file(output_path, metadata)
					
				return tags
		except Exception as e:
			logger.error(f"Błąd: {str(e)}")
			return []

	def predict_with_clip(self, image_path, labels):
		image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
		text = clip.tokenize(labels).to(self.device)

		with torch.no_grad():
			logits_per_image, _ = self.model(image, text)
			probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

		results = []
		for label, prob in zip(labels, probs):
			results.append({
				"tag": label,
				"probability": float(prob)
			})

		return sorted(results, key=lambda x: x["probability"], reverse=True)

	
	def load_images(self, images_path):
		image_files = [
			os.path.join(images_path, f) for f in os.listdir(images_path)
			if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
			]
		images = [self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device) for img_path in image_files]
		return image_files, torch.cat(images)
	
	
	def search_images(self, query, images_path, top_k=5):
		image_files, image_tensors = self.load_images(images_path)
		
		# Wektory zapytania
		text_inputs = clip.tokenize([query]).to(self.device)
		
		with torch.no_grad():
			image_features = self.model.encode_image(image_tensors).float()
			text_features = self.model.encode_text(text_inputs).float()
		
		# Normalizacja wektorów
		image_features /= image_features.norm(dim=-1, keepdim=True)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		
		# Obliczanie podobieństwa kosinusowego
		similarities = (text_features @ image_features.T).squeeze(0).cpu().numpy()
		
		# Posortowanie wyników
		sorted_indices = np.argsort(similarities)[::-1][:top_k]
		results = [(image_files[idx], similarities[idx]) for idx in sorted_indices]
		
		return results
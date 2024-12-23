# -*- coding: utf-8 -*-
"""
Część AI do automatycznego tagowania obrazów z obsługą modelu CLIP.
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
from .logger import get_logger
from tensorflow.keras.applications import MobileNetV2, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import torch
import clip

logger = get_logger(__name__)


class ImageTagger:
	def __init__(self, model_name="InceptionV3"):
		self.model_name = model_name
		
		if model_name == "MobileNetV2":
			self.model = MobileNetV2(weights="imagenet")
			self.image_target_size = (224, 224)
			logger.info("MobileNetV2 model loaded")
		elif model_name == "InceptionV3":
			self.model = InceptionV3(weights="imagenet")
			self.image_target_size = (299, 299)
			logger.info("InceptionV3 model loaded")
		elif model_name == "CLIP":
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
			self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
			logger.info("CLIP model loaded")
		else:
			raise ValueError(f"Unsupported model: {model_name}")
	
	def tag_image(self, image_path=None, output_file=None, top_k=5, labels=None):
		try:
			logger.debug("Przetwarzanie obrazu...")
			
			if self.model_name == "CLIP":
				tags = self.predict_with_clip(image_path, labels)
			else:
				image_array = self.load_and_preprocess_image(image_path)
				logger.debug("Przewidywanie tagów...")
				tags = self.predict_tags(image_array, top_k=top_k)
			
			logger.info(f"[{self.model_name}] Znalezione tagi dla {image_path}: \n{tags}")
			
			if output_file:
				metadata = {
					"file": image_path,
					"tags": tags
				}
				save_metadata_to_file(output_file, metadata)
			
			return tags
		
		except Exception as e:
			logger.error(f"Błąd: {str(e)}")
			return []
	
	def load_and_preprocess_image(self, image_path):
		"""
		Ładuje obraz i przetwarza go do formatu odpowiedniego dla modelu.
		"""
		if not os.path.exists(image_path):
			raise FileNotFoundError(f"Plik {image_path} nie istnieje.")
		
		img = load_img(image_path, target_size=self.image_target_size)
		img_array = img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		return preprocess_input(img_array)
	
	def predict_tags(self, image_array, top_k=5):
		"""
		Przewiduje tagi dla obrazu na podstawie modelu TensorFlow.
		"""
		predictions = self.model.predict(image_array)
		decoded_predictions = decode_predictions(predictions, top=top_k)
		tags = [{"tag": pred[1], "probability": float(pred[2])} for pred in decoded_predictions[0]]
		return tags
	
	def predict_with_clip(self, image_path, labels):
		"""
		Przewiduje tagi dla obrazu przy użyciu modelu CLIP.
		"""
		if not labels:
			labels = ["MEME", "cat", "dog", "television", "other"]
		
		image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
		text = clip.tokenize(labels).to(self.device)
		
		with torch.no_grad():
			# image_features = self.model.encode_image(image)
			# text_features = self.model.encode_text(text)
			logits_per_image, _ = self.model(image, text)
			probs = logits_per_image.softmax(dim=-1).cpu().numpy()
		tags = [{"tag": label, "probability": float(prob)} for label, prob in zip(labels, probs[0])]
		return sorted(tags, key=lambda x: x["probability"], reverse=True)


def save_metadata_to_file(output_path, metadata):
	"""
	Zapisuje metadane w formacie JSON.
	"""
	# if os.path.exists(output_path):
	# 	with open(output_path, 'r') as json_file:
	# 		data = json.load(json_file)
	# 		data.append(metadata)
	# 	with open(output_path, 'w') as json_file:
	# 		json.dump(data, json_file, indent=4)
	# 		logger.debug(f"Metadane dopisane do pliku: {output_path}")
	# else:
	with open(output_path, 'w') as json_file:
		json.dump([metadata], json_file, indent=4)
		logger.info(f"Metadane zapisane do pliku: {output_path}")


if __name__ == "__main__":
	image_tagger_mobilenet = ImageTagger(model_name="MobileNetV2")
	image_tagger_mobilenet.tag_image("..\\tests_image_tagger\\images\\meme.jpg")
	
	image_tagger_inception = ImageTagger(model_name="InceptionV3")
	image_tagger_inception.tag_image("..\\tests_image_tagger\\images\\meme.jpg")
	
	# Przykład użycia CLIP
	labels = ["meme", "person", "selfie", "house", 'car', "other", "friends",
	          "family", "winners", "winner", "losers", "loser", "victory", "success"
	          "instruction", "people", "group", "team", "teamwork", "together", "fun", "landscape", "screenshot"]
	image_tagger_clip = ImageTagger(model_name="CLIP")
	image_tagger_clip.tag_image("..\\tests_image_tagger\\images\\meme.jpg", labels=labels)

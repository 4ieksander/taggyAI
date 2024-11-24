# -*- coding: utf-8 -*-
"""
Część AI do automatycznego tagowania obrazów.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
from logger import get_logger
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

logger = get_logger(__name__)

# import tensorflow as tf
#
# print("Dostępne urządzenia:")
# print(tf.config.list_physical_devices('GPU'))

class ImageTagger:
	def __init__(self):
		self.model = MobileNetV2(weights="imagenet")
	
	def tag_image(self, image_path=None, output_file=None, top_k=5):
		try:
			logger.debug("Przetwarzanie obrazu...")
			image_array = self.load_and_preprocess_image(image_path)
			logger.debug("Przewidywanie tagów...")
			tags = self.predict_tags(image_array, top_k=top_k)
			logger.info(f"Znalezione tagi: {tags}")
			
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


	def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
		"""
		Ładuje obraz i przetwarza go do formatu odpowiedniego dla modelu.
		"""
		if not os.path.exists(image_path):
			raise FileNotFoundError(f"Plik {image_path} nie istnieje.")
		
		img = load_img(image_path, target_size=target_size)
		img_array = img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		return preprocess_input(img_array)
	
	
	def predict_tags(self, image_array, top_k=5):
		"""
		Przewiduje tagi dla obrazu na podstawie modelu.
		"""
		predictions = self.model.predict(image_array)
		decoded_predictions = decode_predictions(predictions, top=top_k)
		tags = [{"tag": pred[1], "probability": float(pred[2])} for pred in decoded_predictions[0]]
		return tags


def save_metadata_to_file(output_path, metadata):
	"""
	Zapisuje metadane w formacie JSON.
	"""
	if os.path.exists(output_path):
		with open(output_path, 'r') as json_file:
			data = json.load(json_file)
			data.append(metadata)
	with open(output_path, 'w') as json_file:
		json.dump(data, json_file, indent=4)
	logger.info(f"Metadane zapisane do pliku: {output_path}")

if __name__ == "__main__":
	image_tagger = ImageTagger()
	image_tagger.tag_image("D:\zdjecia\mieszkanie\IMG-20240609-WA0007.jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\20220717_175535.jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\Screenshot_2022-04-13-09-26-11-83.png")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\20240406_141423.jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\20240921_062820 (1).jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\IMG-4761.jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\IMG_20201209_052142088.jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\inbound3693818213132920604.jpg")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\Kopia 2024-06-04 01-55-05.mkv")
	image_tagger.tag_image("D:\zdjecia\\taggy_test\\VID_133791018_135152_282.mp4")
	

# main('D:\zdjecia\mieszkanie\IMG-20240609-WA0008.jpg')


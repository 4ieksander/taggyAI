import taggy.utils.logger as logger
from taggy.utils.image_tagger import ImageTagger
# import taggy.tests_image_tagger.images

if __name__ == "__main__":
	logger.show_example_logs()
	images = ["meme.jpg", "podium.jpg", "SS_tekstu.png"]
	mobilenet = ImageTagger(model_name="MobileNetV2")
	inception = ImageTagger(model_name="InceptionV3")

	
	clip = ImageTagger(model_name="CLIP")
	labels = ["meme", "person", "selfie", "house", 'car', "other", "friends",
	          "family", "winner", "loser", "instruction", "how_to", "people", "group", "team",
	          "teamwork", "together", "fun", "landscape", "screenshot", "web_site", "text", "graphic", "chart", "table",
	          "diagram", "drawing", "plot", "image", "photo", "picture", "photograph", "screenshot", "document",]
	
	labels_categories = ["screenshot", "documents", "photo", "persons", "mail", "graphics", "delivery", "text", "unknown", "other", "meme"]

	
	
	for image in images:
		img_path = f"images/{image}"
		mobilenet.tag_image(image_path=img_path, output_file=f"{img_path}.mobilebet.json")
		inception.tag_image(img_path, output_file=f"{img_path}.inception.json")
		clip.tag_image(img_path, labels=labels, output_file=f"{img_path}.clip.json")
		clip.tag_image(img_path, labels=labels_categories, output_file=f"{img_path}.clip2.json")



	
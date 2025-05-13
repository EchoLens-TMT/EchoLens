import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption from image
def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to speak the caption
def speak_caption(caption):
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()

# Main function
def process_image(image_path):
    caption = generate_caption(image_path)
    print(f"Caption: {caption}")
    speak_caption(caption)

# Example image path (change this)
image_path = "/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/src/image/mQcKcyt3Nb25vMYBSbb47aQ9Kw.jpg"
process_image(image_path)

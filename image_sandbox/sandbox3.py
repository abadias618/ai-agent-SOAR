from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv    
load_dotenv()

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
api_key = os.getenv("HF")
print("api_key",api_key)



client = InferenceClient(
    provider="hf-inference",
    api_key=api_key,
)
image = client.text_to_image(
    "This is the computer site.  You know, half a dozen PCs, couple of Macs, a printer table, the desk at which you write that interactive fiction game instead of study.  An internet router hums quietly in the corner.  The door is northeast.",
    model="nerijs/pixel-art-xl",
)

# You can access the image with PIL.Image for example
import io
from PIL import Image
#image = Image.open(io.BytesIO(image_bytes))
image.save("./my_image4.png")
print("done",image)
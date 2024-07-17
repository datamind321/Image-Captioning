import streamlit as st 
from PIL import Image
import torch 
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from gtts import gTTS

device = 'cuda' if torch.cuda.is_available() else 'cpu'





model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning').to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

  
def get_caption(model,image_processor,tokenizer,image_path):
  image = Image.open(image_path)
  
  #processing the image
  img = image_processor(image,return_tensors='pt').to(device)
  
  # gteneratimg caption
  output = model.generate(**img)
  
  # decode the output
  caption = tokenizer.batch_decode(output,skip_special_tokens=True)[0]

  return caption




st.title('Vision Transformers (ViT) in Image Captioning Using Pretrained ViT Models')


uploaded_image = st.file_uploader('Upload an Image',type=['png','jpg','jpeg'])

if uploaded_image is not None:
#   image = Image.open(uploaded_image)
  st.image(uploaded_image)
  caption = get_caption(model,image_processor,tokenizer,uploaded_image)
  st.header(caption)
  read_caption = gTTS(caption,lang='en',slow=True)
  read_caption.save('caption.mp3')
  st.audio('caption.mp3',autoplay=True)
else:
    st.error('No Image Uploaded !')
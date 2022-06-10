import requests  # type: ignore
import streamlit as st  # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageOps  # type: ignore
import io
import os
import random

API_URL = "https://labs.dev.bigml.io/andromeda/"
API_USERNAME = os.getenv("BIGML_USERNAME")
API_KEY = os.getenv("BIGML_API_KEY")
API_AUTH = f"username={API_USERNAME};api_key={API_KEY}"
FONT = ImageFont.truetype("img/roboto.ttf", 25)
MODEL = "deepnet/s02lWCyZMJ6qvaI4nBEw81zw6Iv"

HEALTHY_CLASSES = ["Blueberry leaf", "Peach leaf", "Raspberry leaf", "Strawberry leaf", "Tomato leaf",
                   "Bell_pepper leaf", "Soyabean leaf", "Apple leaf", "Cherry leaf", "grape leaf"]
DISEASE_CLASSES = ["Tomato leaf yellow virus", "Tomato Septoria leaf spot", "Corn leaf blight",
                   "Potato leaf early blight", "Tomato mold leaf", "Tomato leaf bacterial spot",
                   "Squash Powdery mildew leaf", "Bell_pepper leaf spot", "Potato leaf late blight",
                   "Tomato leaf mosaic virus", "Tomato leaf late blight", "Tomato Early blight leaf",
                   "Apple rust leaf", "Apple Scab Leaf", "grape leaf black rot", "Corn rust leaf",
                   "Corn Gray leaf spot", "Tomato two spotted spider mites leaf"]


def resize(img, width):
    """ Resize an imge to a given width maintaining aspect ratio """
    percent = width / float(img.size[0])
    return img.resize((width, int((float(img.size[1]) * float(percent)))))


def detection(uploaded_file):
    """Apply object detection to the uploaded file"""
    source_response = requests.post(
        f"{API_URL}source?{API_AUTH}",
        files={"file": ("plant_image", uploaded_file)}
    )
    source = source_response.json()["resource"]
    data = {"model": MODEL, "input_data": {"000002": source}}
    response = requests.post(f"{API_URL}prediction?{API_AUTH}", json=data)
    boxes = response.json()["prediction"].get("000000", [])
    if len(boxes) == 0:
        print("WARNING: No bounding boxes found")
    return boxes


def draw_predictions(pil_image, boxes):
    """ Draw BigML predictions in the image, adding a black border too """
    w, h = pil_image.size
    draw = ImageDraw.Draw(pil_image)
    for box in boxes:
        label, xmin,ymin, xmax, ymax, confidence = box
        draw.rectangle(((xmin*w, ymin*h), (xmax*w, ymax*h)), width=9, outline="#eee")
        draw.text(
            (xmin*w+20, ymin*h+random.randint(10, 40)),
            f"{label}: {str(confidence)[:3]}", font=FONT,  fill="#eee"
        )
    return ImageOps.expand(pil_image ,border=50,fill='black')


def gen_message(boxes):
    """ Generate output message for predictions """
    labels = set([box[0] for box in boxes])
    healthy = labels.intersection(set(HEALTHY_CLASSES))
    diseases = labels.intersection(set(DISEASE_CLASSES))
    if len(diseases) > 0:        
        st.warning(f"Your plants needs a doctor!. Found **{','.join(diseases)}**!")
    else:
        st.success(f"Your plants have good health!. Found **{','.join(healthy)}**!")



st.set_page_config(
    layout="wide",
    page_title="Plant Disease Detection",
    page_icon="🌱",
)

# Sidebar information
description = """ Detect leafs from different kinds of plants and
diagnose common **diseases** in them.  """
image = Image.open('img/BigML.png')
st.sidebar.image(image, width=100)
st.sidebar.write(description)
st.sidebar.write("Powered by [BigML](bigml.com)")

# Page title
st.title("🌱 BigML Plant Disease Detection")
classes = "HEALTHY:\n"
for leaf in HEALTHY_CLASSES:
    classes += f"- {leaf}\n"
classes += "\nDISEASES:\n"
for leaf in DISEASE_CLASSES:
    classes += f"- {leaf}\n"

with st.expander("Model trained to find the following classes: "):
    st.write(classes) 

# File uploader
msg = "Pleanse upload your plant image."
uploaded_file = st.file_uploader(msg, type=["png ", "jpg", "jpeg"])

# Example images
examples = {
    "Example Plant 1": "img/plant3.jpg",
    "Example Plant 2": "img/plant8.jpg",
    "Example Plant 3": "img/39-late-blight-mold.jpg",    
    "Example Plant 4": "img/601fbee78c62e32a76e768a92ee40193.jpg"
}
with st.expander("Or select one of these examples:"):
    option = st.selectbox('Choose one', examples.keys(),index=0)
    clicked = st.button("Select example")
    if clicked:
        uploaded_file = open(examples[option], 'rb')
    for col, title in zip(st.columns(4), examples.keys()):    
        col.image(Image.open(examples[title]), caption=title)

# Prediction Output
if uploaded_file:
    st.subheader("Detection result")
    with st.spinner('Diagnose in progress. Please wait...'):
        boxes = detection(uploaded_file)
        uploaded_image = resize(Image.open(uploaded_file), 1000)
        output_image = draw_predictions(uploaded_image, boxes)
        gen_message(boxes)
        st.image(output_image, width=700)

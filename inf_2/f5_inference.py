# streamlit_app.py
import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from f4_model import Face_v2

# --- Load model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('./models/face_multitask_v3_att2_1763066628.pth', map_location=device, weights_only=False)
model.to(device)
model.eval()

AGE_MAP = {
    0: "0-3",
    1: "4-19",
    2: "20-39",
    3: "40-69",
    4: "70+"
}

EMOTION_MAP = {
    0: "Surprise",
    1: "Fear + Disgust",
    2: "Happiness",
    3: "Sadness + Anger",
    4: "Neutral"
}

GENDER_MAP = {
    0: "Male",
    1: "Female",
    2: "Unsure"
}

RACE_MAP = {
    0: "Caucasian",
    1: "African-American",
    2: "Asian"
}


# --- Prediction function ---
def predict(image):
    # Convert to RGB
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Resize to model input
    input_img = cv2.resize(img, (128, 128))
    input_tensor = torch.tensor(input_img.transpose(2,0,1), dtype=torch.float32).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    # Model forward
    with torch.no_grad():
        landmarks, emotion, race, gender, age = model(input_tensor)
    
    # Convert landmarks back to original image size
    landmarks = landmarks.cpu().numpy().reshape(-1, 2)
    landmarks[:,0] *= w
    landmarks[:,1] *= h
    
    # Get class predictions
    emotion = torch.argmax(emotion, dim=1).item()
    race = torch.argmax(race, dim=1).item()
    gender = torch.argmax(gender, dim=1).item()
    age = torch.argmax(age, dim=1).item()
    
    return {
        'landmarks': landmarks,
        'emotion': emotion,
        'race': race,
        'gender': gender,
        'age': age
    }

# --- Visualization ---
def show_prediction(image, prediction):
    img = np.array(image)
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    lms = prediction['landmarks']
    plt.scatter(lms[:,0], lms[:,1], c='r', s=40)
    plt.title(f"Emotion: {EMOTION_MAP[prediction['emotion']]}, Race: {RACE_MAP[prediction['race']]}, "
              f"Gender: {GENDER_MAP[prediction['gender']]}, Age: {AGE_MAP[prediction['age']]}")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# --- Streamlit app ---
st.title("Face Attribute Prediction")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prediction = predict(image)
    show_prediction(image, prediction)

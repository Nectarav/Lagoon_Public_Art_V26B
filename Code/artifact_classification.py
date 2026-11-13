import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import joblib
import io

# to open run:
# streamlit run artifact_classification.py

# load the model
model_data = joblib.load('artifact_classifier_model.pkl')

model = models.resnet18(pretrained = False)

model.fc = torch.nn.Linear(model.fc.in_features, len(model_data['class_to_idx']))
model.load_state_dict(model_data['model_state_dict'])
model.eval()

## everything ready to be used!

idx_to_class = {v: k for k, v in model_data['class_to_idx'].items()}

transform = transforms.Compose([
    transforms.Resize((3008,2000)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title('Artifact Classifier')

st.markdown('Upload an artifact image and the model will predict what type of artifact it is.')

uploaded_file = st.file_uploader('Upload an image', type = ['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption = 'Uploaded Image', use_container_width=True)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = idx_to_class[predicted.item()]

    st.success(f'Predicted Artifact: **{class_name}**')
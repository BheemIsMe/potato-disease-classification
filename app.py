import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding',False)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./models/1')
    return model

model = load_model()

st.write("""
    # Potato Disease Classification
""")


file = st.file_uploader("Please upload an flower image",type=["jpg","png","jpeg"])


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img,dtype='uint8')
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predicted_class, confidence = predict(model,image)
    st.success(f"{predicted_class}, {confidence}")


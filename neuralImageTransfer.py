import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# Title
st.title("🎨 Neural Style Transfer Web App")

# Upload images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

# Image preprocessing
def load_image(image_file):
    img = Image.open(image_file).resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Load VGG19
@st.cache_resource
def load_model():
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    return model

vgg = load_model()

content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']

def get_features(model, image):
    outputs = [model.get_layer(name).output for name in style_layers + [content_layer]]
    feature_model = tf.keras.Model([model.input], outputs)
    return feature_model(image)

def compute_loss(gen_img, content_img, style_img):
    gen_features = get_features(vgg, gen_img)
    content_features = get_features(vgg, content_img)
    style_features = get_features(vgg, style_img)

    content_loss = tf.reduce_mean((gen_features[-1] - content_features[-1])**2)

    style_loss = 0
    for g, s in zip(gen_features[:-1], style_features[:-1]):
        style_loss += tf.reduce_mean((g - s)**2)

    return content_loss + style_loss

# Run style transfer
if content_file and style_file:
    st.image(content_file, caption="Content Image")
    st.image(style_file, caption="Style Image")

    if st.button("Generate Styled Image"):
        content_image = load_image(content_file)
        style_image = load_image(style_file)

        generated_image = tf.Variable(content_image, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=5.0)

        st.write("Processing... Please wait ⏳")

        for i in range(50):  # keep small for speed
            with tf.GradientTape() as tape:
                loss = compute_loss(generated_image, content_image, style_image)
            grad = tape.gradient(loss, generated_image)
            optimizer.apply_gradients([(grad, generated_image)])

        result = generated_image.numpy().squeeze()
        result = np.clip(result, 0, 255).astype('uint8')

        st.image(result, caption="Stylized Output 🎨")
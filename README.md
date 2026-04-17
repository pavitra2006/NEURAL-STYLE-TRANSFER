# NEURAL-STYLE-TRANSFER

COMPANY : CODTECH IT SOLUTIONS NAME : PAVITRA SAVARAPU INTERN ID : CTIS7186 DOMAIN : Artificial Intelligence DURATION : 6 WEEKS MENTOR : NEELA SANTOSH

*DESCRIPTION*:
## Neural Style Transfer Project Description

The neuralImageTransfer.py is a Streamlit-based neural style transfer application that combines a content image with an artistic style image to create a stylized output using deep learning.

### What the task does
- Provides a web interface for uploading two images: a content image and a style image
- Applies neural style transfer to blend the content structure with the style's artistic elements
- Generates and displays a new image that retains the content's shapes while adopting the style's colors and textures
- Allows users to view the original images and the stylized result

### Tools and architecture
- Streamlit: builds the interactive web app, manages image uploads, displays results, and handles the generation process
- TensorFlow/Keras: loads VGG19 model for feature extraction, performs gradient-based optimization to minimize content and style loss
- Image processing: PIL for loading and resizing images, NumPy for array manipulations

### Libraries used
- `streamlit`
- `tensorflow`
- `numpy`
- `PIL` (Pillow)

### How it works
1. User uploads a content image and a style image via Streamlit
2. Images are resized to 224x224 and preprocessed for VGG19
3. VGG19 extracts features from multiple layers for content (block5_conv2) and style (block1-3_conv1)
4. A generated image (initialized as content) is optimized using Adam optimizer
5. Loss function combines content loss (MSE between generated and content features) and style loss (Gram matrix differences)
6. After 50 iterations, the optimized image is postprocessed and displayed

### Why it is useful
This project demonstrates advanced computer vision techniques for artistic image manipulation, combining convolutional neural networks with optimization. It showcases how deep learning can create novel visual content, useful for digital art, design, and creative applications. The interactive web interface makes it accessible for non-technical users to experiment with style transfer.

*OUTPUT*:
https://github.com/user-attachments/assets/28a5c11a-4c5e-4e04-a364-ca5509e36663

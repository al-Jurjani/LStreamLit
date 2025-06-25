import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

def main():
    st.title("CIFAR-10 Image Classification")
    st.write("Upload an image to classify it using a pre-trained model.")

    file = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image to match CIFAR-10 input size and format
        image = image.convert('RGB')  # Ensure the image is in RGB format
        resized_image = image.resize((32, 32))  # Resize to match CIFAR-10 input size
        image_array = np.array(resized_image) / 255.0  # Normalize the image
        image_array = image_array.reshape((1, 32, 32, 3))  # Reshape for the model input - 1 image, 32x32 pixels, 3 colour channels

        # Load the pre-trained model and make predictions
        model = tf.keras.models.load_model('cifar10_model.h5')  # Load the pre-trained model
        predictions = model.predict(image_array)  # Make predictions

        cifar10_classes = [
            'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
            'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
        ] # From the CIFAR-10 dataset documentation
        
        # matplotlib for plotting the predictions
        st.write("Predictions:")
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('CIFAR-10 Class Predictions')

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        st.warning("Please upload an image file to proceed.")
        return

if __name__ == "__main__":
    main()
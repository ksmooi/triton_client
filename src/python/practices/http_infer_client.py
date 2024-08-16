import numpy as np
import json
import tritonclient.http as httpclient
from PIL import Image

# Load class labels from JSON file
with open("images/imagenet-simple-labels.json", "r") as f:
    class_labels = json.load(f)

# Create a Triton client
triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)

# Define the model name and input/output names
model_name = "mobilenetv4_hybrid-plan"
input_name = "input"
output_name = "output"

# Create input data with the correct shape
batch_size = 4  # Adjusted batch size to be within the allowed limit
input_shape = (batch_size, 3, 384, 384)  # Shape for a single input

# Generate 4 different input data arrays
# Load the images
image_paths = ["images/ship.jpg", "images/cat.jpg", "images/truck.jpg", "images/tiger.jpg"]
image_list = []
for path in image_paths:
    image = Image.open(path)
    image = image.resize((384, 384))  # Resize the image to match the input shape
    image = np.array(image)
    # Preprocess the image using ImageNet normalization
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Subtract mean and divide by standard deviation
    image_list.append(image)

# Transpose the dimensions to match the expected shape
input_data = np.stack(image_list).astype(np.float32)    # shape: (4, 384, 384, 3)
input_data = np.transpose(input_data, (0, 3, 1, 2))     # shape: (4, 3, 384, 384)

# Create input and output objects
inputs = [httpclient.InferInput(input_name, input_data.shape, "FP32")]
outputs = [httpclient.InferRequestedOutput(output_name, binary_data=True)]

# Set the input data
inputs[0].set_data_from_numpy(input_data)

try:
    # Perform inference
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    # Get the output data
    output_data = results.as_numpy(output_name)

    # Print the shape and type of the output data
    print("Output data shape:", output_data.shape)
    print("Output data type:", output_data.dtype)
    print()

    # Example: If the output is class probabilities, you might want to find the class with the highest probability
    if output_data.ndim == 2:  # Assuming output is [batch_size, num_classes]
        for i in range(output_data.shape[0]):
            predicted_class = np.argmax(output_data[i])
            probability = output_data[i][predicted_class]
            print(f"Image {i+1}:")
            print(f"Predicted class: {class_labels[predicted_class]}")
            print(f"Probability: {probability:.4f}")
            print()
    else:
        print("Output data:", output_data)
    
except utils.InferenceServerException as e:
    print(f"Inference failed: {e}")
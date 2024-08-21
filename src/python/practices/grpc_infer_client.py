import numpy as np
import json
import tritonclient.grpc as grpcclient
from PIL import Image
import argparse
import sys

# Function to load class labels from JSON file
def load_labels(label_file):
    with open(label_file, "r") as f:
        return json.load(f)

# Function to preprocess an image
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path)
    image = image.resize((input_shape[2], input_shape[3]))  # Resize the image to match the input shape
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Subtract mean and divide by standard deviation
    image = np.transpose(image, (2, 0, 1))  # Change dimensions to match (3, H, W)
    return image.astype(np.float32)

# Main inference function
def infer(model_name, image_paths, label_file="images/imagenet-simple-labels.json"):
    # Load class labels
    class_labels = load_labels(label_file)

    # Create a Triton client
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

    # Define input and output names
    input_name = "input"
    output_name = "output"

    # Assume all images have the same dimensions as the first image
    image_list = [preprocess_image(image_path, (1, 3, 384, 384)) for image_path in image_paths]
    input_data = np.stack(image_list)  # Combine images into a batch
    input_shape = input_data.shape

    # Create input and output objects
    inputs = [grpcclient.InferInput(input_name, input_shape, "FP32")]
    outputs = [grpcclient.InferRequestedOutput(output_name)]

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

        # Process and display the results
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

    except grpcclient.utils.InferenceServerException as e:
        print(f"Inference failed: {e}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using a Triton server.")
    parser.add_argument("-m", "--model_name", required=True, help="Name of the model to use for inference.")
    parser.add_argument("-i", "--image_paths", action='append', required=True, help="Path to an input image file. Use multiple -i options for multiple images.")
    args = parser.parse_args()

    if len(args.image_paths) == 0:
        print("Error: At least one image path must be provided.")
        sys.exit(1)

    infer(args.model_name, args.image_paths)

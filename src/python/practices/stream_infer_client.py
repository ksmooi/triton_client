import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient
import tritonclient.utils as utils
import json

# Load class labels from JSON file
with open("images/imagenet-simple-labels.json", "r") as f:
    class_labels = json.load(f)

# Create a Triton client with verbose logging enabled
triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

# Define the model name and input/output names
model_name = "mobilenetv4_hybrid-plan"
input_name = "input"
output_name = "output"

# Define the maximum batch size (should match the model configuration)
max_batch_size = 4

# Create input data with the correct shape
batch_size = 4  # Adjusted batch size to be within the allowed limit
if batch_size > max_batch_size:
    raise ValueError(f"Batch size {batch_size} exceeds the maximum batch size {max_batch_size}")

input_shape = (batch_size, 3, 384, 384)  # Shape for a single input

# Generate 4 different input data arrays
# Load the images
image_paths = ["images/ship.jpg", "images/cat.jpg", "images/truck.jpg", "images/tiger.jpg"]
input_data = []
for path in image_paths:
    image = Image.open(path)
    image = image.resize((384, 384))  # Resize the image to match the input shape
    image = np.array(image)
    # Preprocess the image using ImageNet normalization
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Subtract mean and divide by standard deviation
    input_data.append(image)

input_data = np.stack(input_data).astype(np.float32)
input_data = np.transpose(input_data, (0, 3, 1, 2))  # Transpose the dimensions to match the expected shape

# Create input and output objects
inputs = [grpcclient.InferInput(input_name, input_data.shape, "FP32")]
outputs = [grpcclient.InferRequestedOutput(output_name)]

# Set the input data
inputs[0].set_data_from_numpy(input_data)

# Define a callback function to handle responses
def callback(result, error):
    if error:
        print(f"Inference failed: {error}")
    else:
        try:
            output_data = result.as_numpy(output_name)
            # Print the shape and type of the output data
            print("Output data shape:", output_data.shape)
            print("Output data type:", output_data.dtype)

            # Example: If the output is class probabilities, you might want to find the class with the highest probability
            if output_data.ndim == 2:  # Assuming output is [batch_size, num_classes]
                for i in range(output_data.shape[0]):
                    predicted_class = np.argmax(output_data[i])
                    probability = output_data[i][predicted_class]
                    print(f"Image {i+1}:")
                    print(f"Predicted class: {class_labels[predicted_class]}")
                    print(f"Probability: {probability:.4f}")
            else:
                print("Output data:", output_data)
        except Exception as e:
            print(f"Error processing the result: {e}")

# Start the streaming inference
try:
    stream = triton_client.start_stream(callback=callback)
    if stream is None:
        raise RuntimeError("Failed to create a streaming inference context")

    with stream:
        # Send the inference request
        stream.async_stream_infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # You can send more requests here if needed
        # For example:
        # stream.async_stream_infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Wait for all responses to be received
        stream.complete()
except utils.InferenceServerException as e:
    print(f"Failed to start streaming inference: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Note: The stream will automatically close when exiting the 'with' block
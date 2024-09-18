import numpy as np
import json
from PIL import Image
import argparse
import sys

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

class TritonInferenceClient:
    """
    A client to perform image inference using the Triton Inference Server.

    This class supports both gRPC and HTTP protocols for communication with the
    Triton server and provides methods to preprocess images, send inference requests,
    and handle responses.

    Attributes:
        protocol (str): Communication protocol ('grpc' or 'http').
        model_name (str): Name of the model to use for inference.
        server_url (str): URL of the Triton server.
        server_port (str): Port of the Triton server.
        client: Triton Inference Server client instance.
        input_name (str): Name of the model's input tensor.
        output_name (str): Name of the model's output tensor.
        verbose (bool): Enables verbose logging if set to True.
    """

    def __init__(self, protocol, model_name, server_url='localhost', server_port=None):
        """
        Initializes the TritonInferenceClient.

        Args:
            protocol (str): Protocol to use ('grpc' or 'http').
            model_name (str): Name of the model to use for inference.
            server_url (str, optional): URL of the Triton server. Defaults to 'localhost'.
            server_port (str, optional): Port of the Triton server. Defaults are '8001' for gRPC and '8000' for HTTP.
        """
        self.protocol = protocol.lower()
        self.model_name = model_name
        self.server_url = server_url
        self.server_port = server_port
        self.client = None
        self.input_name = 'input'
        self.output_name = 'output'
        self.verbose = False  # Set to True to enable verbose logging
        self.init_client()

    def init_client(self):
        """
        Initializes the Triton client based on the specified protocol.

        Raises:
            ValueError: If an unsupported protocol is specified.
        """
        if self.protocol == 'grpc':
            # Set default port if not provided
            if self.server_port is None:
                self.server_port = '8001'  # Default gRPC port
            # Construct the server URL
            url = f"{self.server_url}:{self.server_port}"
            # Initialize the gRPC client
            self.client = grpcclient.InferenceServerClient(
                url=url, verbose=self.verbose)
        elif self.protocol == 'http':
            # Set default port if not provided
            if self.server_port is None:
                self.server_port = '8000'  # Default HTTP port
            # Construct the server URL
            url = f"{self.server_url}:{self.server_port}"
            # Initialize the HTTP client
            self.client = httpclient.InferenceServerClient(
                url=url, verbose=self.verbose)
        else:
            raise ValueError("Protocol must be either 'grpc' or 'http'")

    def load_labels(self, label_file):
        """
        Loads class labels from a JSON file.

        Args:
            label_file (str): Path to the JSON file containing class labels.
        """
        with open(label_file, 'r') as f:
            self.class_labels = json.load(f)

    def preprocess_image(self, image_path, input_shape):
        """
        Preprocesses an input image for model inference.

        Steps:
            - Opens the image and converts it to RGB.
            - Resizes the image to match the model's expected input dimensions.
            - Normalizes pixel values to [0, 1].
            - Applies mean and standard deviation normalization.
            - Transposes the image to match the model's input format (C, H, W).

        Args:
            image_path (str): Path to the input image file.
            input_shape (tuple): Expected input shape (batch_size, channels, height, width).

        Returns:
            np.ndarray: The preprocessed image as a NumPy array.
        """
        # Open the image file and ensure it's in RGB format
        image = Image.open(image_path).convert('RGB')
        # Resize the image to match the model's expected input dimensions
        image = image.resize((input_shape[2], input_shape[3]))
        # Convert the image to a NumPy array and normalize pixel values to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0

        # Define mean and standard deviation arrays for normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Normalize the image
        image = (image - mean) / std

        # Transpose the image to (channels, height, width)
        image = np.transpose(image, (2, 0, 1))

        # Ensure the final output is float32
        return image.astype(np.float32)

    def infer(self, image_paths, label_file='images/imagenet-simple-labels.json'):
        """
        Performs inference on the provided images.

        Args:
            image_paths (list): List of paths to input image files.
            label_file (str, optional): Path to the JSON file containing class labels.
        """
        # Load class labels
        self.load_labels(label_file)

        # Define the input shape (assuming all images have the same dimensions)
        # For this example, we assume input shape is (1, 3, 384, 384)
        # TODO: Dynamically obtain input shape from the model metadata
        input_shape = (1, 3, 384, 384)

        # Preprocess images
        image_list = [self.preprocess_image(
            image_path, input_shape) for image_path in image_paths]

        # Stack images into a batch
        input_data = np.stack(image_list)
        # Update input shape based on the actual batch size
        input_shape = input_data.shape

        # Create input and output objects for the inference request
        if self.protocol == 'grpc':
            # Create input and output tensors for gRPC
            inputs = [grpcclient.InferInput(
                self.input_name, input_shape, "FP32")]
            outputs = [grpcclient.InferRequestedOutput(self.output_name)]
            # Set the input data
            inputs[0].set_data_from_numpy(input_data)
        else:  # HTTP protocol
            # Create input and output tensors for HTTP
            inputs = [httpclient.InferInput(
                self.input_name, input_shape, "FP32")]
            outputs = [httpclient.InferRequestedOutput(self.output_name)]
            # Set the input data, using binary data for efficiency
            inputs[0].set_data_from_numpy(input_data, binary_data=True)

        try:
            # Perform inference
            results = self.client.infer(
                model_name=self.model_name, inputs=inputs, outputs=outputs)

            # Retrieve the output data
            output_data = results.as_numpy(self.output_name)

            # Check the output data dimensions
            if output_data.ndim == 2:  # Expected shape: [batch_size, num_classes]
                for i in range(output_data.shape[0]):
                    # Get the predicted class index
                    predicted_class = np.argmax(output_data[i])
                    # Get the probability of the predicted class
                    probability = output_data[i][predicted_class]
                    print(f"Image {i+1}:")
                    print(f"Predicted class: {self.class_labels[predicted_class]}")
                    print(f"Probability: {probability:.4f}")
                    print()
            else:
                # Handle unexpected output shapes
                print("Output data:", output_data)

        except InferenceServerException as e:
            # Print the inference error
            print(f"Inference failed: {e}")

# Usage:
# cd ~/GitHub/nvidia/triton_client/src/python
# python practices/xinfer_client.py -p grpc -m mobilenetv4_conv_small-plan -i images/tiger.png
# python practices/xinfer_client.py -p http -m mobilenetv4_conv_small-plan -i images/cat.jpg
#
if __name__ == "__main__":
    # Command-line interface for the TritonInferenceClient
    parser = argparse.ArgumentParser(description="Perform inference using a Triton server.")
    parser.add_argument("-p", "--protocol", required=True, choices=['grpc', 'http'],
                        help="Protocol to use for communication with the Triton server.")
    parser.add_argument("-m", "--model", required=True,
                        help="Name of the model to use for inference.")
    parser.add_argument("-i", "--image", action='append', required=True,
                        help="Path to an input image file. Use multiple -i options for multiple images.")
    parser.add_argument("-u", "--url", default='localhost',
                        help="URL of the Triton server (default: localhost).")
    parser.add_argument("-sp", "--server_port", default=None,
                        help="Port of the Triton server. Defaults are 8001 for gRPC and 8000 for HTTP.")

    args = parser.parse_args()

    # Ensure at least one image path is provided
    if len(args.image) == 0:
        print("Error: At least one image path must be provided.")
        sys.exit(1)

    # Initialize the TritonInferenceClient with the parsed arguments
    client = TritonInferenceClient(protocol=args.protocol,
                                   model_name=args.model,
                                   server_url=args.url,
                                   server_port=args.server_port)

    # Perform inference on the provided images
    client.infer(args.image)

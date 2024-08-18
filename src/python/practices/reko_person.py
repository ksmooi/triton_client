import argparse
import cv2
import numpy as np
import tritonclient.grpc as grpcclient

class PersonReID:
    """
    PersonReID is a class that encapsulates the functionality for performing 
    person re-identification using a model hosted on Triton Inference Server.
    
    Attributes:
        model_name (str): The name of the model to use for inference.
        image_height (int): The height to resize input images to.
        image_width (int): The width to resize input images to.
        triton_client (InferenceServerClient): Triton client for making inference requests.
    """

    def __init__(self, model_name, image_height=256, image_width=128, triton_url="localhost:8001"):
        """
        Initializes the PersonReID class with the model name, image dimensions, and Triton server URL.
        
        Args:
            model_name (str): The name of the model to use for inference.
            image_height (int): The height to resize input images to.
            image_width (int): The width to resize input images to.
            triton_url (str): The URL of the Triton Inference Server.
        """
        self.model_name = model_name
        self.image_height = image_height
        self.image_width = image_width
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)

    def preprocess_image(self, image_path):
        """
        Preprocesses the input image for inference.
        
        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image tensor.
        """
        # Load the image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Image not found: {image_path}")
        
        # Convert from BGR to RGB
        original_image = original_image[:, :, ::-1]
        
        # Resize the image
        image_resized = cv2.resize(original_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to float32 and transpose to (C, H, W)
        image_transposed = image_resized.astype("float32").transpose(2, 0, 1)
        
        # Add batch dimension (1, C, H, W)
        image_batch = np.expand_dims(image_transposed, axis=0)
        
        return image_batch

    @staticmethod
    def normalize(nparray, order=2, axis=-1):
        """
        Normalizes a N-D numpy array along the specified axis.
        
        Args:
            nparray (np.ndarray): The array to normalize.
            order (int): The order of the norm (default is 2, i.e., L2 norm).
            axis (int): The axis along which to normalize.

        Returns:
            np.ndarray: Normalized array.
        """
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): The first embedding.
            embedding2 (np.ndarray): The second embedding.

        Returns:
            float: The cosine similarity score.
        """
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity

    def infer_embedding(self, image):
        """
        Performs inference on the preprocessed image to obtain the embedding.
        
        Args:
            image (np.ndarray): The preprocessed image tensor.

        Returns:
            np.ndarray: The normalized embedding output by the model.
        """
        # Prepare the input for the inference request
        inputs = [grpcclient.InferInput("input", image.shape, "FP32")]
        
        # Set the input data
        inputs[0].set_data_from_numpy(image)
        
        # Define the output we want to retrieve
        output = grpcclient.InferRequestedOutput("output")
        
        # Perform the inference request
        results = self.triton_client.infer(self.model_name, inputs, outputs=[output])
        
        # Get the embedding and normalize it
        embedding = results.as_numpy("output").flatten()
        embedding = self.normalize(embedding, axis=-1)
        
        return embedding

    def compare_images(self, image_path1, image_path2):
        """
        Compares two images by computing the similarity of their embeddings.
        
        Args:
            image_path1 (str): Path to the first image.
            image_path2 (str): Path to the second image.

        Returns:
            float: The similarity score between the two images.
        """
        # Preprocess both images
        image1 = self.preprocess_image(image_path1)
        image2 = self.preprocess_image(image_path2)
        
        # Get embeddings for both images
        embedding1 = self.infer_embedding(image1)
        embedding2 = self.infer_embedding(image2)
        
        # Calculate similarity between the two embeddings
        similarity = self.calculate_similarity(embedding1, embedding2)
        
        return similarity

if __name__ == "__main__":
    # Argument parser to get command-line arguments
    parser = argparse.ArgumentParser(description="Person Re-Identification using Triton Inference Server")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("image_path1", type=str, help="Path to the first image")
    parser.add_argument("image_path2", type=str, help="Path to the second image")
    parser.add_argument("--height", type=int, default=256, help="Height to resize the image to")
    parser.add_argument("--width", type=int, default=128, help="Width to resize the image to")
    args = parser.parse_args()

    # Create an instance of the PersonReID class
    reid = PersonReID(model_name=args.model_name, image_height=args.height, image_width=args.width)

    # Compare the two images and print the similarity score
    similarity = reid.compare_images(args.image_path1, args.image_path2)
    print(f"Similarity between the two images: {similarity:.4f}")

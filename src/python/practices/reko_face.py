import tritonclient.grpc as grpcclient
import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine
import argparse

class FaceRecognitionModel:
    """
    A class to interact with a face recognition model served by Triton Inference Server.

    Attributes:
    ----------
    model_name : str
        The name of the model on the Triton server.
    triton_client : grpcclient.InferenceServerClient
        The Triton Inference Server client used to communicate with the server.

    Methods:
    -------
    preprocess_image(image_path):
        Preprocesses an image for input into the model.
    
    get_embedding(image):
        Sends a preprocessed image to the Triton server and retrieves the embedding.
    
    compare_faces(image_path1, image_path2):
        Compares the embeddings of two images and returns the cosine similarity.
    
    display_images(image_path1, image_path2):
        Displays the two images side by side.
    """

    def __init__(self, model_name, server_url="localhost:8001"):
        """
        Initializes the FaceRecognitionModel with the model name and server URL.

        Parameters:
        ----------
        model_name : str
            The name of the model on the Triton server.
        server_url : str, optional
            The URL of the Triton server (default is "localhost:8001").
        """
        self.model_name = model_name
        self.triton_client = grpcclient.InferenceServerClient(url=server_url)
    
    def preprocess_image(self, image_path):
        """
        Preprocesses an image by resizing, converting to RGB, normalizing, and converting to tensor.

        Parameters:
        ----------
        image_path : str
            The file path of the image to be processed.

        Returns:
        -------
        np.ndarray
            The preprocessed image ready for model input.
        """
        # Load image from the given path
        img = cv2.imread(image_path)
        # Resize image to 112x112 as required by the model
        img = cv2.resize(img, (112, 112))
        # Convert image from BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Transpose image dimensions from HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Convert the image to a PyTorch tensor and add a batch dimension
        img = torch.from_numpy(img).unsqueeze(0).float()
        # Normalize the image: img = (img / 255 - 0.5) / 0.5
        img.div_(255).sub_(0.5).div_(0.5)
        # Convert the tensor back to a NumPy array
        img = img.numpy()
        return img
    
    def get_embedding(self, image):
        """
        Sends a preprocessed image to the Triton server and retrieves the face embedding.

        Parameters:
        ----------
        image : np.ndarray
            The preprocessed image in NumPy array format.

        Returns:
        -------
        np.ndarray
            The embedding vector representing the face in the image.
        """
        # Create input object for the image with the appropriate shape and data type
        inputs = []
        inputs.append(grpcclient.InferInput("input.1", image.shape, "FP32"))
        inputs[0].set_data_from_numpy(image)
        
        # Create output object for the embedding with the appropriate name
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("683"))
        
        # Perform inference on the Triton server and retrieve the results
        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        # Flatten the result to get the embedding vector
        embedding = results.as_numpy("683").flatten()
        return embedding
    
    @staticmethod
    def compute_cosine_similarity(embedding1, embedding2):
        """
        Computes the cosine similarity between two embedding vectors.

        Parameters:
        ----------
        embedding1 : np.ndarray
            The first embedding vector.
        embedding2 : np.ndarray
            The second embedding vector.

        Returns:
        -------
        float
            The cosine similarity between the two embeddings.
        """
        # Calculate cosine similarity between the two embeddings
        return 1 - cosine(embedding1, embedding2)

    def compare_faces(self, image_path1, image_path2):
        """
        Compares the face embeddings of two images and returns their similarity.

        Parameters:
        ----------
        image_path1 : str
            The file path of the first image.
        image_path2 : str
            The file path of the second image.

        Returns:
        -------
        float
            The cosine similarity between the two face embeddings.
        """
        # Preprocess both images
        image1 = self.preprocess_image(image_path1)
        image2 = self.preprocess_image(image_path2)
        
        # Retrieve embeddings for both images
        embedding1 = self.get_embedding(image1)
        embedding2 = self.get_embedding(image2)
        
        # Compute and return the cosine similarity between the embeddings
        similarity = self.compute_cosine_similarity(embedding1, embedding2)
        return similarity

    def display_images(self, image_path1, image_path2):
        """
        Displays the two images side by side using OpenCV.

        Parameters:
        ----------
        image_path1 : str
            The file path of the first image.
        image_path2 : str
            The file path of the second image.
        """
        # Load the two images
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        # Display the images using OpenCV
        cv2.imshow("Image 1", img1)
        cv2.imshow("Image 2", img2)
        cv2.waitKey(0)  # Wait for a key press to close the windows
        cv2.destroyAllWindows()


def main(model_name, image_path1, image_path2):
    """
    Main function to compare face embeddings of two images and display them.

    Parameters:
    ----------
    model_name : str
        The name of the model on the Triton server.
    image_path1 : str
        The file path of the first image.
    image_path2 : str
        The file path of the second image.
    """
    # Create an instance of the FaceRecognitionModel
    face_recognition_model = FaceRecognitionModel(model_name)
    
    # Compare the face embeddings of the two images and print the similarity
    similarity = face_recognition_model.compare_faces(image_path1, image_path2)
    print(f"Cosine similarity between the two images: {similarity:.4f}")
    
    # Display the images for visual inspection
    face_recognition_model.display_images(image_path1, image_path2)


if __name__ == "__main__":
    # Argument parser for command-line input
    parser = argparse.ArgumentParser(description='Compare face embeddings of two images using Triton client and OpenCV')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('image_path1', type=str, help='Path to the first input image')
    parser.add_argument('image_path2', type=str, help='Path to the second input image')
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.model_name, args.image_path1, args.image_path2)

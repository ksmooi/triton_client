import argparse
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from PIL import Image
import torchvision.transforms as transforms

class FacialLandmarkDetector:
    """
    Class to handle facial landmark detection using Triton Inference Server and OpenCV.
    
    Attributes:
        model_name (str): The name of the model deployed on Triton Inference Server.
        triton_client (InferenceServerClient): The client used to communicate with Triton Inference Server.
        mean (np.ndarray): Mean values for image normalization.
        std (np.ndarray): Standard deviation values for image normalization.
        resize (transforms.Resize): Resize transform for preprocessing.
        to_tensor (transforms.ToTensor): Transform to convert image to tensor.
        normalize (transforms.Normalize): Transform to normalize the image tensor.
    """

    def __init__(self, model_name):
        """
        Initialize the detector with the model name and setup Triton client.

        Args:
            model_name (str): Name of the model to be used for inference.
        """
        self.model_name = model_name
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

        # Preprocessing configuration
        self.mean = np.asarray([0.485, 0.456, 0.406])
        self.std = np.asarray([0.229, 0.224, 0.225])
        self.resize = transforms.Resize([56, 56])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def preprocess(self, image):
        """
        Preprocess the input image before sending it to the model.

        Args:
            image (np.ndarray): The original image in BGR format.

        Returns:
            np.ndarray: The preprocessed image ready for inference.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # TODO: image transpose to CHW format
                
        image = image.unsqueeze(0)
        return image.numpy()

    def infer(self, preprocessed_image):
        """
        Perform inference on the preprocessed image using Triton Inference Server.

        Args:
            preprocessed_image (np.ndarray): The preprocessed image.

        Returns:
            np.ndarray: The output from the model containing landmark predictions.
        """
        inputs = [grpcclient.InferInput("input", preprocessed_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(preprocessed_image)
        outputs = [grpcclient.InferRequestedOutput("output")]

        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        return results.as_numpy("output")

    def reproject_landmark(self, landmark, bbox):
        """
        Reproject the predicted landmarks back onto the original image scale.

        Args:
            landmark (np.ndarray): The predicted landmarks.
            bbox (list): The bounding box of the face in the original image.

        Returns:
            np.ndarray: The reprojected landmarks.
        """
        landmark = landmark.copy()
        x1, x2, y1, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        landmark[:, 0] = landmark[:, 0] * w + x1
        landmark[:, 1] = landmark[:, 1] * h + y1
        return landmark

    def postprocess(self, output, bbox, orig_image):
        """
        Postprocess the model output to display landmarks on the original image.

        Args:
            output (np.ndarray): The raw output from the model.
            bbox (list): The bounding box of the face in the original image.
            orig_image (np.ndarray): The original image.

        Returns:
            np.ndarray: The image with landmarks drawn on it.
        """
        landmark = output.reshape(-1, 2)
        landmark = self.reproject_landmark(landmark, bbox)
        for (x, y) in landmark:
            cv2.circle(orig_image, (int(x), int(y)), 2, (0, 255, 0), -1)
        return orig_image

    def detect_and_display(self, image_path):
        """
        Perform the complete detection process and display the annotated image.

        Args:
            image_path (str): The path to the input image.
        """
        orig_image = cv2.imread(image_path)
        if orig_image is None:
            print(f"Error: Unable to load image at {image_path}")
            return

        # Preprocess the image
        preprocessed_image = self.preprocess(orig_image)

        # Perform inference
        output_data = self.infer(preprocessed_image)

        # Define a dummy bounding box for full image (adjust as needed)
        bbox = [0, orig_image.shape[1], 0, orig_image.shape[0]]

        # Postprocess and display the image
        annotated_image = self.postprocess(output_data[0], bbox, orig_image)
        cv2.imshow('Facial Landmark Detection', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(args):
    """
    Main function to initialize the FacialLandmarkDetector and run detection.

    Args:
        args: Command line arguments containing model name and image path.
    """
    detector = FacialLandmarkDetector(model_name=args.model_name)
    detector.detect_and_display(args.image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Landmark Detection using Triton Inference Server')
    parser.add_argument('-m', '--model_name', required=True, help='Name of the model to use')
    parser.add_argument('-i', '--image_path', required=True, help='Path to the image file')
    args = parser.parse_args()
    
    main(args)

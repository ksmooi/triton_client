import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import cv2
from collections import namedtuple

# Define the Face data structure
# Face is a named tuple to store detected face data, including bounding box (bbox), keypoints (kps), and confidence score (score).
Face = namedtuple('Face', ['bbox', 'kps', 'score'])

# Image preprocessing function
def preprocess_image(image: np.ndarray, input_size=(640, 640)):
    """
    Preprocess the input image to prepare it for model inference.

    Parameters:
    - image (np.ndarray): The original image loaded from disk.
    - input_size (tuple): The desired input size for the model, typically (640, 640).

    Returns:
    - np.ndarray: The preprocessed image ready for inference.
    - dict: Metadata containing information needed for post-processing.
    """
    # Resize the image while keeping the aspect ratio
    shape = image.shape[:2]
    ratio = min(input_size[0] / shape[0], input_size[1] / shape[1])
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = (input_size[1] - new_unpad[0]) / 2, (input_size[0] - new_unpad[1]) / 2
    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding to make the image the target size
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Normalize the image (scale pixel values to [0, 1])
    image_normalized = image_padded.astype(np.float32) / 255.0
    
    # Convert BGR to RGB as the model expects the input in RGB format
    image_rgb = image_normalized[..., ::-1]
    
    # Change format from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) as expected by the model
    image_chw = image_rgb.transpose((2, 0, 1))
    
    # Store metadata for later use in post-processing (like restoring the original aspect ratio)
    meta_data = {
        'ratio': ratio,
        'dw': dw,
        'dh': dh
    }
    
    # Return the preprocessed image and metadata
    return np.ascontiguousarray(np.array([image_chw])), meta_data

# Post-processing function
def postprocess_predictions(predictions, meta_data, score_threshold=0.25, iou_threshold=0.50):
    """
    Post-process the model's predictions to extract and filter bounding boxes, keypoints, and scores.

    Parameters:
    - predictions (np.ndarray): Raw output from the model.
    - meta_data (dict): Metadata from preprocessing used to restore original dimensions.
    - score_threshold (float): Minimum confidence score to consider a detection.
    - iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).

    Returns:
    - list[Face]: A list of detected faces with bounding boxes, keypoints, and confidence scores.
    """
    # Reshape predictions to a more usable format (n, 8400, 20)
    predictions = np.transpose(predictions, (0, 2, 1))
    predictions = np.ascontiguousarray(predictions)

    faces = []
    for i, pred in enumerate(predictions):
        # Split the predictions into bounding box (bbox), confidence score (score), and keypoints (kps)
        bbox, score, kps = np.split(pred, [4, 5], axis=1)
        ratio, dw, dh = meta_data['ratio'], meta_data['dw'], meta_data['dh']

        # Convert bounding box format from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
        new_ratio = 1 / ratio
        x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x_min = (x_center - (width / 2) - dw) * new_ratio
        y_min = (y_center - (height / 2) - dh) * new_ratio
        x_max = (x_center + (width / 2) - dw) * new_ratio
        y_max = (y_center + (height / 2) - dh) * new_ratio
        bbox = np.stack((x_min, y_min, x_max, y_max), axis=1)

        # Convert keypoints from (x, y, score) to (x, y) format
        for j in range(kps.shape[1] // 3):
            kps[:, j * 3] = (kps[:, j * 3] - dw) * new_ratio
            kps[:, j * 3 + 1] = (kps[:, j * 3 + 1] - dh) * new_ratio

        # Filter predictions by confidence score threshold
        indices_above_threshold = np.where(score > score_threshold)[0]
        bbox = bbox[indices_above_threshold]
        score = score[indices_above_threshold]
        kps = kps[indices_above_threshold]

        # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
        nms_indices = cv2.dnn.NMSBoxes(bbox.tolist(), score.ravel().tolist(), score_threshold, iou_threshold)
        if len(nms_indices) > 0:
            bbox = bbox[nms_indices.flatten()]
            score = score[nms_indices.flatten()]
            kps = kps[nms_indices.flatten()]
            
            # Create Face objects and add them to the list of detected faces
            for j in range(len(bbox)):
                faces.append(Face(bbox=bbox[j], kps=kps[j], score=score[j]))

    return faces

# Draw bounding boxes and keypoints on the image
def draw_faces(image, faces):
    """
    Draw bounding boxes, keypoints, and confidence scores on the image.

    Parameters:
    - image (np.ndarray): The original image on which to draw.
    - faces (list[Face]): A list of detected faces with bounding boxes, keypoints, and confidence scores.

    Returns:
    - np.ndarray: The image with drawn bounding boxes, keypoints, and confidence scores.
    """
    for face in faces:
        bbox = face.bbox.astype(int)  # Convert bounding box coordinates to integer
        kps = face.kps.reshape(-1, 3)  # Reshape keypoints into (x, y, score) format
        score = face.score[0]  # Extract the confidence score
        
        # Draw the bounding box around the detected face
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Draw keypoints (eyes, nose, mouth) on the face
        for kp in kps:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        
        # Draw the confidence score above the bounding box
        label = f"Confidence: {score:.2f}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Main function to perform face detection
def detect_faces(model_name, conf_threshold, iou_threshold, image_path):
    """
    Perform face detection on an image using a YOLOv8 model hosted on a Triton Inference Server.

    Parameters:
    - model_name (str): The name of the model to use.
    - conf_threshold (float): The confidence threshold for detection.
    - iou_threshold (float): The IoU threshold for Non-Maximum Suppression (NMS).
    - image_path (str): The path to the input image.

    This function handles the full detection process: loading the image, preprocessing, inference, post-processing, and drawing the results.
    """
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

    # Load and preprocess the image
    original_image = cv2.imread(image_path)
    input_data, meta_data = preprocess_image(original_image)

    # Create input and output objects for Triton
    inputs = [grpcclient.InferInput("images", input_data.shape, "FP32")]
    outputs = [grpcclient.InferRequestedOutput("output0")]

    # Set input data
    inputs[0].set_data_from_numpy(input_data)

    # Perform inference with the Triton server
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    output_data = results.as_numpy("output0")

    # Post-process the model's output to extract faces
    faces = postprocess_predictions(output_data, meta_data, score_threshold=conf_threshold, iou_threshold=iou_threshold)

    # Draw the detected faces on the original image
    image_with_faces = draw_faces(original_image, faces)

    # Display the image with the detected faces
    cv2.imshow("Detected Faces", image_with_faces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Entry point for the script
if __name__ == "__main__":
    """
    Parse command-line arguments and call the detect_faces function.

    The script accepts the following arguments:
    - --model-name: The name of the model to use.
    - --conf-threshold: The confidence threshold for detection.
    - --iou-threshold: The IoU threshold for NMS.
    - --image-path: The path to the input image.
    """
    parser = argparse.ArgumentParser(description="Face detection using Triton Inference Server")
    parser.add_argument('-m', '--model-name', type=str, required=True, default="yolov8n-face-plan", help="Name of the model to use (default: yolov8n-face-plan)")
    parser.add_argument('-c', '--conf-threshold', type=float, default=0.25, help="Confidence threshold for detection (default: 0.25)")
    parser.add_argument('-u', '--iou-threshold', type=float, default=0.60, help="IoU threshold for NMS (default: 0.60)")
    parser.add_argument('-i', '--image-path', type=str, required=True, default="images/face_01.jpg", help="Path to the input image (default: images/face_01.jpg)")
    args = parser.parse_args()

    detect_faces(args.model_name, args.conf_threshold, args.iou_threshold, args.image_path)

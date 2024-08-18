import argparse
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
import cv2

# Function to parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for the object detection script.
    
    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Object detection using Triton Inference Server")
    parser.add_argument('-m', '--model-name', type=str, required=True, default="yolov8m-det-plan",
                        help="Name of the model to use (default: yolov8m-det-plan)")
    parser.add_argument('-c', '--conf-threshold', type=float, default=0.5,
                        help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument('-u', '--iou-threshold', type=float, default=0.4,
                        help="IoU threshold for NMS (default: 0.4)")
    parser.add_argument('-i', '--image-path', type=str, required=True, default="images/tiger.jpg",
                        help="Path to the input image (default: images/tiger.jpg)")
    return parser.parse_args()

# Image preprocessing function
def preprocess_image(image_path, target_size=(640, 640)):
    """
    Preprocesses the input image by resizing and normalizing it.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired size of the output image (width, height).

    Returns:
        image_np (numpy.ndarray): Preprocessed image array ready for model input.
    """
    image = Image.open(image_path)  # Open the image
    image = image.resize(target_size)  # Resize the image to match model input size
    image_np = np.array(image).astype(np.float32)  # Convert image to numpy array and float32
    image_np = image_np / 255.0  # Normalize pixel values to [0, 1]
    image_np = np.transpose(image_np, (2, 0, 1))  # Convert image to CHW format (channels, height, width)
    return image_np

# Post-processing function to apply NMS and interpret model output
def postprocess_output(input_image, output, conf_threshold=0.5, iou_threshold=0.4):
    """
    Processes the output of the model by applying Non-Maximum Suppression (NMS) and filtering results.

    Args:
        input_image (numpy.ndarray): Original input image.
        output (numpy.ndarray): Model output array.
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): NMS threshold for suppressing overlapping boxes.

    Returns:
        final_boxes (list): List of bounding boxes after NMS.
        final_scores (list): List of confidence scores after NMS.
        final_class_ids (list): List of class IDs after NMS.
    """
    outputs = np.transpose(np.squeeze(output[0]))  # Squeeze and transpose the output to match expected shape
    rows = outputs.shape[0]

    boxes, scores, class_ids = [], [], []

    img_height, img_width = input_image.shape[:2]  # Get image dimensions
    x_factor = img_width / 640  # Factor to rescale bounding box coordinates (width)
    y_factor = img_height / 640  # Factor to rescale bounding box coordinates (height)

    for i in range(rows):
        class_scores = outputs[i][4:]  # Extract class scores from the output
        max_score = np.amax(class_scores)  # Get the maximum score for a class

        if max_score >= conf_threshold:  # Only consider detections above the confidence threshold
            class_id = np.argmax(class_scores)  # Get the class ID with the highest score
            x_center, y_center, width, height = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Convert center coordinates to top-left and bottom-right coordinates
            x1 = int((x_center - width / 2) * x_factor)
            y1 = int((y_center - height / 2) * y_factor)
            x2 = int((x_center + width / 2) * x_factor)
            y2 = int((y_center + height / 2) * y_factor)

            boxes.append([x1, y1, x2, y2])  # Append bounding box coordinates
            scores.append(max_score)  # Append confidence score
            class_ids.append(class_id)  # Append class ID

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    if len(indices) > 0:
        final_boxes = [boxes[i] for i in indices.flatten()]  # Get final bounding boxes after NMS
        final_scores = [scores[i] for i in indices.flatten()]  # Get final scores after NMS
        final_class_ids = [class_ids[i] for i in indices.flatten()]  # Get final class IDs after NMS
    else:
        final_boxes = []
        final_scores = []
        final_class_ids = []

    return final_boxes, final_scores, final_class_ids

# Draw bounding boxes on the image
def draw_boxes(image, boxes, scores, class_ids, class_labels):
    """
    Draws bounding boxes and labels on the image.

    Args:
        image (numpy.ndarray): Original image to draw on.
        boxes (list): List of bounding boxes.
        scores (list): List of confidence scores for each box.
        class_ids (list): List of class IDs for each box.
        class_labels (list): List of class labels for each class ID.

    Returns:
        image (numpy.ndarray): Image with drawn bounding boxes and labels.
    """
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Draw rectangle (bounding box)
        label_text = f"{class_labels[class_id]}: {score:.2f}"  # Prepare label text with class and score
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Draw label
    return image

# Main function to perform object detection
def detect_objects(model_name, conf_threshold, iou_threshold, image_path):
    """
    Performs object detection on a single image using the Triton Inference Server.

    Args:
        model_name (str): Name of the model to use for inference.
        conf_threshold (float): Confidence threshold for detection.
        iou_threshold (float): NMS threshold for filtering overlapping boxes.
        image_path (str): Path to the input image.
    """
    # Create a Triton client for communication with the inference server
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
    
    # Placeholder class labels, replace with actual labels as needed
    class_labels = [f"Class {i}" for i in range(80)]

    # Preprocess the image for model input
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension to the image

    # Create input and output objects for the Triton server
    inputs = [grpcclient.InferInput("images", input_data.shape, "FP32")]
    outputs = [grpcclient.InferRequestedOutput("output0")]

    # Set the input data
    inputs[0].set_data_from_numpy(input_data)

    # Perform inference using the Triton server
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    output_data = results.as_numpy("output0")  # Get the output from the server

    # Load the original image to draw bounding boxes on
    original_image = cv2.imread(image_path)

    # Post-process the model output to get bounding boxes, scores, and class IDs
    boxes, scores, class_ids = postprocess_output(original_image, output_data, conf_threshold, iou_threshold)

    # Draw the bounding boxes and labels on the original image
    image_with_boxes = draw_boxes(original_image, boxes, scores, class_ids, class_labels)

    # Display the image with detections
    cv2.imshow("Detections", image_with_boxes)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the image window

# Entry point for the script
if __name__ == "__main__":
    args = parse_args()  # Parse the command-line arguments
    detect_objects(args.model_name, args.conf_threshold, args.iou_threshold, args.image_path)  # Perform object detection

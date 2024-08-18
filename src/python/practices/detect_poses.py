import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import cv2
from PIL import Image

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Pose detection using Triton Inference Server")
    parser.add_argument('-m', '--model-name', type=str, required=True, help="Name of the model to use")
    parser.add_argument('-i', '--image-path', type=str, required=True, help="Path to the input image")
    return parser.parse_args()

# Image preprocessing function
def preprocess_image(image_path, target_size=(640, 640)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    return image_np

# Post-processing function
def postprocess_output(output, conf_threshold=0.5):
    output_transposed = output[0].T
    bboxes = output_transposed[:, :4]
    conf = output_transposed[:, 4]
    kpts = output_transposed[:, 5:]

    # Apply confidence threshold
    index = np.argmax(conf)

    # Extract keypoints for the highest confidence detection
    keypoints = kpts[index].reshape((17, 3))
    return bboxes[index], keypoints

# Draw the skeleton
def draw_skeleton(image, keypoints):
    labels = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    skeleton = [
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
    ]

    keypoint_dict = {label: (int(kp[0]), int(kp[1])) for label, kp in zip(labels, keypoints)}

    for joint in skeleton:
        pt1 = keypoint_dict[joint[0]]
        pt2 = keypoint_dict[joint[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    for x, y, _ in keypoints:
        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

    return image

# Main function to perform pose detection
def detect_poses(model_name, image_path):
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
    
    # Preprocess the image
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)
    
    # Create input and output objects for Triton
    inputs = [grpcclient.InferInput("images", input_data.shape, "FP32")]
    outputs = [grpcclient.InferRequestedOutput("output0")]

    # Set input data
    inputs[0].set_data_from_numpy(input_data)

    # Perform inference using Triton
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    output_data = results.as_numpy("output0")

    # Load the original image to draw on
    original_image = cv2.imread(image_path)

    # Post-process the output to get keypoints
    bbox, keypoints = postprocess_output(output_data)

    # Draw skeleton on the original image
    image_with_skeleton = draw_skeleton(original_image, keypoints)

    # Display the image with keypoints and bounding boxes
    cv2.imshow("Detected Poses", image_with_skeleton)  # Corrected variable name
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the image window

# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    detect_poses(args.model_name, args.image_path)

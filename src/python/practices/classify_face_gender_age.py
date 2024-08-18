import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import argparse

class TritonClient:
    def __init__(self, url="localhost:8001"):
        self.client = grpcclient.InferenceServerClient(url=url)

    def infer(self, model_name, input_data):
        inputs = []
        outputs = []
        
        inputs.append(grpcclient.InferInput('data', input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data.astype(np.float32))  # Ensure data is FP32

        outputs.append(grpcclient.InferRequestedOutput('fc1'))

        results = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        return results

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    input_mean = 127.5
    input_std = 128.0
    
    # Normalize the image
    face_blob = cv2.dnn.blobFromImage(img, 1.0/input_std, (96, 96), (input_mean, input_mean, input_mean), swapRB=True)

    return face_blob

def parse_logits(logits):
    assert len(logits) == 3
    gender = np.argmax(logits[:2])
    age = int(np.round(logits[2] * 100))
    return gender, age

def main(args):
    # Preprocess the input image
    input_data = preprocess_image(args.input)
    
    # Create a TritonClient instance
    triton_client = TritonClient()

    # Perform inference on the Triton Inference Server
    results = triton_client.infer(args.model, input_data)

    # Get the inference output
    output = results.as_numpy("fc1")[0]
    print(f"output: {output}")

    # Parse the logits to get gender and age
    gender, age = parse_logits(output)

    # Print the results
    print(f"Gender: {'Male' if gender == 1 else 'Female'}, Age: {age}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify face gender and age using Triton Inference Server')
    parser.add_argument('-m', '--model', required=True, help='Model name')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    args = parser.parse_args()
    main(args)
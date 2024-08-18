import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tritonclient.grpc as grpcclient

class ImageProcessor:
    """
    A class that provides image processing functionalities.
    Methods:
    - preprocess_warpAffine(image, dst_width=640, dst_height=640): Preprocesses the input image using the warpAffine transformation.
    """
    @staticmethod
    def preprocess_warpAffine(image, dst_width=640, dst_height=640):
        """
        Preprocesses the input image using the warpAffine transformation.
        Parameters:
        - image: The input image to be preprocessed.
        - dst_width (optional): The width of the output image after preprocessing. Default is 640.
        - dst_height (optional): The height of the output image after preprocessing. Default is 640.
        Returns:
        - img_pre: The preprocessed image.
        - IM: The inverse transformation matrix used for preprocessing.
        """
        # Calculate the scaling factor for the image
        scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
        
        # Calculate the translation values for centering the image
        ox = (dst_width  - scale * image.shape[1]) / 2
        oy = (dst_height - scale * image.shape[0]) / 2
        
        # Create the transformation matrix
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)
        
        # Apply the warpAffine transformation to the image
        img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        
        # Calculate the inverse transformation matrix
        IM = cv2.invertAffineTransform(M)

        # Convert the preprocessed image to the desired format
        img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
        img_pre = img_pre.transpose(2, 0, 1)[None]
        img_pre = torch.from_numpy(img_pre)
        
        return img_pre, IM


class ObjectDetector:
    @staticmethod
    def iou(box1, box2):
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes.
        Parameters:
        - box1: The coordinates of the first bounding box in the format [x1, y1, x2, y2].
        - box2: The coordinates of the second bounding box in the format [x1, y1, x2, y2].
        Returns:
        - iou: The IoU value between the two bounding boxes.
        """
        def area_box(box):
            """
            Calculates the area of a bounding box.
            Parameters:
            - box: The coordinates of the bounding box in the format [x1, y1, x2, y2].
            Returns:
            - area: The area of the bounding box.
            """
            return (box[2] - box[0]) * (box[3] - box[1])

        # Calculate the coordinates of the intersection rectangle
        left   = max(box1[0], box2[0])
        top    = max(box1[1], box2[1])
        right  = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])

        # Calculate the area of the intersection rectangle
        cross  = max((right-left), 0) * max((bottom-top), 0)

        # Calculate the area of the union rectangle
        union  = area_box(box1) + area_box(box2) - cross

        # Calculate the Intersection over Union (IoU) value
        return cross / union if cross > 0 and union > 0 else 0

    @classmethod
    def NMS(cls, boxes, iou_thres):
        """
        Performs Non-Maximum Suppression (NMS) on a list of bounding boxes.
        Parameters:
        - boxes: The list of bounding boxes to perform NMS on.
        - iou_thres: The IoU threshold for removing overlapping boxes.
        Returns:
        - keep_boxes: The list of bounding boxes after NMS.
        """
        # Initialize a list to keep track of boxes to remove
        remove_flags = [False] * len(boxes)
        keep_boxes = []
        
        # Iterate over each box
        for i, ibox in enumerate(boxes):
            if remove_flags[i]:
                continue
            keep_boxes.append(ibox)
            
            # Check for overlapping boxes with the same label
            for j in range(i + 1, len(boxes)):
                if remove_flags[j] or ibox[5] != boxes[j][5]:
                    continue
                if cls.iou(ibox, boxes[j]) > iou_thres:
                    remove_flags[j] = True
        
        return keep_boxes

    @classmethod
    def postprocess(cls, pred, conf_thres=0.25, iou_thres=0.45):
        """
        Performs post-processing on the predicted bounding boxes.
        Parameters:
        - pred: The predicted bounding boxes.
        - conf_thres (optional): The confidence threshold for filtering out low-confidence predictions. Default is 0.25.
        - iou_thres (optional): The IoU threshold for NMS. Default is 0.45.
        Returns:
        - boxes: The final list of bounding boxes after post-processing.
        """
        # Initialize an empty list to store the bounding boxes
        boxes = []

        # Iterate over each item in the prediction
        for item in pred[0]:
            # Extract the coordinates and label information from the item
            cx, cy, w, h = item[:4]
            label = item[4:-32].argmax()
            confidence = item[4 + label]

            # Check if the confidence is below the threshold, skip if true
            if confidence < conf_thres:
                continue

            # Calculate the left, top, right, and bottom coordinates of the bounding box
            left, top = cx - w * 0.5, cy - h * 0.5
            right, bottom = cx + w * 0.5, cy + h * 0.5

            # Append the bounding box information to the list
            boxes.append([left, top, right, bottom, confidence, label, *item[-32:]])

        # Sort the bounding boxes based on confidence in descending order
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        # Perform Non-Maximum Suppression (NMS) on the bounding boxes
        return cls.NMS(boxes, iou_thres)


class MaskProcessor:
    @staticmethod
    def crop_mask(masks, boxes):
        """
        Crop the masks based on the bounding boxes.
        Parameters:
        - masks: The input masks.
        - boxes: The bounding boxes.
        Returns:
        - cropped_masks: The cropped masks.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    @classmethod
    def process_mask(cls, protos, masks_in, bboxes, shape, upsample=False):
        """
        Process the masks based on the prototypes and bounding boxes.
        Parameters:
        - protos: The prototypes.
        - masks_in: The input masks.
        - bboxes: The bounding boxes.
        - shape: The shape of the output masks.
        - upsample (optional): Whether to upsample the masks. Default is False.
        Returns:
        - masks: The processed masks.
        """
        c, mh, mw = protos.shape  # Get the shape of the 'protos' tensor
        ih, iw = shape  # Get the shape of the 'shape' tensor
        
        # Perform matrix multiplication, sigmoid activation, and reshape the 'masks_in' tensor
        masks = (masks_in.float() @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  

        downsampled_bboxes = bboxes.clone()  # Create a copy of the 'bboxes' tensor
        downsampled_bboxes[:, [0, 2]] *= mw / iw  # Adjust the bounding box coordinates based on the width ratio
        downsampled_bboxes[:, [1, 3]] *= mh / ih  # Adjust the bounding box coordinates based on the height ratio

        masks = cls.crop_mask(masks, downsampled_bboxes)  # Crop the masks based on the bounding boxes using the 'crop_mask' function
        if upsample:
            # Upsample the masks using bilinear interpolation if 'upsample' is True
            masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  
        
        return masks.gt_(0.5)  # Threshold the masks by comparing each element to 0.5 and return the result


class Visualizer:
    @staticmethod
    def hsv2bgr(h, s, v):
        # Convert HSV color to BGR color
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0: r, g, b = v, t, p
        elif h_i == 1: r, g, b = q, v, p
        elif h_i == 2: r, g, b = p, v, t
        elif h_i == 3: r, g, b = p, q, v
        elif h_i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q

        return int(b * 255), int(g * 255), int(r * 255)

    @classmethod
    def random_color(cls, id):
        # Generate a random color based on the object id
        h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
        s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
        return cls.hsv2bgr(h_plane, s_plane, 1)

    @classmethod
    def draw_masks_and_boxes(cls, img, masks, boxes, IM):
        """
        Draw masks and bounding boxes on the input image.
        Parameters:
        - img: The input image.
        - masks: The masks to be drawn.
        - boxes: The bounding boxes to be drawn.
        - IM: The transformation matrix for resizing the masks.
        Returns:
        - img: The image with masks and bounding boxes drawn.
        """
        h, w = img.shape[:2]
        # Iterate over each mask
        for i, mask in enumerate(masks):
            mask = mask.cpu().numpy().astype(np.uint8)
            # Resize the mask using the transformation matrix IM
            mask_resized = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR)
            
            label = int(boxes[i][5])
            # Generate a random color based on the object label
            color = np.array(cls.random_color(label))
            
            # Create a colored mask using the generated color
            colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
            # Apply the resized mask to the colored mask
            masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

            # Get the indices where the mask is equal to 1
            mask_indices = mask_resized == 1
            # Blend the original image with the masked colored mask
            img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)

        # Iterate over each bounding box
        for obj in boxes:
            left, top, right, bottom = map(int, obj[:4])
            confidence, label = obj[4], int(obj[5])
            # Generate a random color based on the object label
            color = cls.random_color(label)
            # Draw a rectangle around the bounding box
            cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
            caption = f"{label} {confidence:.2f}"
            text_w, text_h = cv2.getTextSize(caption, 0, 1, 2)[0]
            # Draw a filled rectangle as the background for the caption
            cv2.rectangle(img, (left - 3, top - 33), (left + text_w + 10, top), color, -1)
            # Draw the caption text
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

        return img

class TritonClient:
    """
    A client for interacting with the Triton Inference Server.
    """

    def __init__(self, url="localhost:8001"):
        """
        Initialize the TritonClient with the server URL.
        Parameters:
        - url (optional): The URL of the Triton Inference Server. Default is "localhost:8001".
        """
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def infer(self, model_name, input_data):
        """
        Perform inference on the Triton Inference Server.
        Parameters:
        - model_name: The name of the deployed model on the server.
        - input_data: The input data for inference.
        Returns:
        - results: The inference results.
        - input_data_copy: A copy of the input data.
        """
        inputs = [grpcclient.InferInput("images", input_data.shape, "FP32")]
        outputs = [
            grpcclient.InferRequestedOutput("output0"),
            grpcclient.InferRequestedOutput("output1")
        ]
        inputs[0].set_data_from_numpy(input_data)
        results = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
        # Return both results and a copy of input_data
        return results, input_data.copy()  

def main(args):
    # Read the input image
    img = cv2.imread(args.input)
    
    # Display the shape of the image
    print("shape of source image:", img.shape)
    
    # Preprocess the image and get the transformation matrix
    img_pre, IM = ImageProcessor.preprocess_warpAffine(img)
    input_data = img_pre.numpy()

    # Ensure the input data has a batch dimension
    if len(input_data.shape) == 3:
        input_data = np.expand_dims(input_data, axis=0)

    # Create a TritonClient instance
    triton_client = TritonClient()

    # Perform inference on the Triton Inference Server
    results, input_data_copy = triton_client.infer(args.model, input_data)

    # Get the inference outputs
    output0 = results.as_numpy("output0")
    output1 = results.as_numpy("output1")

    # Convert the outputs to torch tensors
    output0 = torch.from_numpy(output0.copy()).transpose(-1, -2)
    output1 = torch.from_numpy(output1[0].copy())

    # Perform post-processing on the outputs
    pred = ObjectDetector.postprocess(output0)
    pred = torch.from_numpy(np.array(pred).reshape(-1, 38))

    # Process the masks based on the prototypes and bounding boxes
    masks = MaskProcessor.process_mask(output1, pred[:, 6:], pred[:, :4], img_pre.shape[2:], True)

    # Adjust the bounding box coordinates based on the transformation matrix
    boxes = np.array(pred[:,:6])
    lr, tb = boxes[:, [0, 2]], boxes[:,[1, 3]]
    boxes[:,[0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:,[1, 3]] = IM[1][1] * tb + IM[1][2]

    # Draw masks and bounding boxes on the input image
    img_result = Visualizer.draw_masks_and_boxes(img, masks, boxes, IM)

    # Display the segmentation result
    cv2.imshow("Segmentation Result", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation with Triton Inference Server")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name deployed on Triton Inference Server")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image path")
    args = parser.parse_args()
    
    main(args)

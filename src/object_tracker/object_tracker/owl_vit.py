import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Change this to your image topic
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.pil_image = None
        self.cv_image = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rgb_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        self.cv_image = rgb_image  # Store OpenCV image for later use
        self.pil_image = PILImage.fromarray(cv_image)
        cv2.imshow("Image Window", self.cv_image)
        cv2.waitKey(100)  # Wait for a key press for 1 millisecond

class TextSubscriber(Node):
    def __init__(self):
        super().__init__('text_subscriber')
        self.subscription = self.create_subscription(
            String,
            'text_topic',  # Change this to your text topic
            self.text_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.text = None
        self.entering = False

    def text_callback(self, msg):
        self.text = msg.data
        self.entering = True
        print(self.text)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    text_subscriber = TextSubscriber()

    # Initialize processor and model once
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(text_subscriber)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if text_subscriber.entering and image_subscriber.pil_image is not None:
                text_subscriber.entering = False
                image = image_subscriber.pil_image
                target_sizes = torch.tensor([image.size[::-1]])
                Label = text_subscriber.text
                text = f"a photo of a {Label}"
                
                # Prepare inputs for the model
                inputs = processor(text=text, images=image, return_tensors="pt")
                threshold = 0.02

                # Run inference
                outputs = model(**inputs)

                # Post-process the outputs
                results=processor.post_process_object_detection(outputs, target_sizes=target_sizes,threshold=threshold)
                i=0
                boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
                for box, score, label in zip(boxes, scores, labels):
                     box = [round(i, 2) for i in box.tolist()]
                     print(f"Detected {Label} with confidence {round(score.item(), 3)} at location {box}")

                # Draw the bounding boxes and labels on the image
                # for box, score, label in zip(boxes, scores, labels):

                #     if score > threshold:  # Only display boxes with confidence > 0.5
                #         x_min, y_min, x_max, y_max = [int(coord) for coord in box]
                #         cv2.rectangle(image_subscriber.cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                #         cv2.putText(image_subscriber.cv_image, f"{Label}: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw the bounding box with the largest score
                if len(scores) > 0:
                    max_score_index = scores.argmax()
                    box = boxes[max_score_index]
                    score = scores[max_score_index]
                    label = labels[max_score_index]
                    box = [round(i, 2) for i in box.tolist()]
                    x_min, y_min, x_max, y_max = [int(coord) for coord in box]
                    cv2.rectangle(image_subscriber.cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image_subscriber.cv_image, f"{Label}: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the image with bounding boxes
                cv2.imshow("Detected Objects", image_subscriber.cv_image)
                cv2.waitKey(500)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        image_subscriber.destroy_node()
        text_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
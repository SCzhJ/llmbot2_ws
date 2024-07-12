
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
from .submodules.owl_vit import OWLViT, ImageSubscriber, TextSubscriber
from .submodules.sam import SAM
from .submodules.XMem_track import XMemTrack
import numpy as np

    

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    text_subscriber = TextSubscriber()

    # Initialize processor and model once
    owlvit = OWLViT()
    sam = SAM()
    xmem = XMemTrack()

    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(text_subscriber)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if text_subscriber.entering and image_subscriber.pil_image is not None:
                text_subscriber.entering = False
                Label = text_subscriber.text
                text = f"a photo of a {Label}"

                image = image_subscriber.pil_image

                results = owlvit.predict(image, text_subscriber.text)

                box, score, label = owlvit.get_best_box(results)
                mask = sam.predict_from_boxes(image_subscriber.cv_image, np.array(box))
                processor, prediction = xmem.initialize(1, mask, image_subscriber.cv_image)
                while rclpy.ok() and not text_subscriber.entering:
                    executor.spin_once(timeout_sec=0.05)
                    processor, prediction = xmem.track(processor, image_subscriber.cv_image)
                    print(prediction.shape)
                    # use cv2 to visualize the mask
                    cv2.imshow("mask", prediction*200)
                    cv2.waitKey(100)
                    time.sleep(0.1)

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
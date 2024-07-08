import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TextPublisher(Node):
    def __init__(self):
        super().__init__('text_publisher')
        self.publisher_ = self.create_publisher(String, 'text_topic', 10)
        timer_period = 0.5  # seconds (adjust as needed)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        input_text = input("Enter text to publish: ")
        msg.data = f"{input_text}"
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: '{msg.data}'")

def main(args=None):
    rclpy.init(args=args)
    text_publisher = TextPublisher()
    try:
        rclpy.spin(text_publisher)
    except KeyboardInterrupt:
        pass
    text_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
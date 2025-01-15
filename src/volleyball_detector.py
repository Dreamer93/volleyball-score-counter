import cv2
import numpy as np
from net_detector import NetDetector, DebugStep

class VolleyballScoreDetector:
    def __init__(self):
        self.net_coordinates = None
        self.court_lines = None
        self.net_detector = NetDetector()

    def load_image(self, image_path):
        """Load and return an image from the given path"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from path: {image_path}")
        print(f"Successfully loaded image with shape: {image.shape}")
        return image

    def detect_net(self, frame):
        """
        Detect volleyball net in the frame
        Returns the coordinates of the net line
        """
        return self.net_detector.detect(frame)

    def detect_court_lines(self, frame):
        """
        Detect court lines based on the net position
        Returns the coordinates of relevant court lines
        """
        # TODO: Implement court line detection
        pass

    def process_frame(self, frame):
        """
        Process a single frame and detect net and court lines
        
        Args:
            frame: Input frame
        """
        debug_step = DebugStep.DENSITY
        if debug_step:
            return self.net_detector.detect(frame, debug_step)
        
        # Normal processing
        net = self.detect_net(frame)
        
        if net is not None:
            self.net_coordinates = net
            self.court_lines = self.detect_court_lines(frame)
        
        self.draw_detections(frame)
        
        return frame

    def draw_detections(self, frame):
        """Draw detected net and court lines on the frame"""
        if self.net_coordinates is not None:
            # Draw net line
            for line in self.net_coordinates:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

if __name__ == "__main__":
    # Create detector instance
    detector = VolleyballScoreDetector()
    
    # Load test image
    image = detector.load_image("./test_image.jpg")
    
    # Process image
    result = detector.process_frame(image)
    
    # Create window
    window_name = "Volleyball Court Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Display result
    cv2.imshow(window_name, result)
    
    # Wait for 'q' key to quit
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows() 
import cv2
import numpy as np
from enum import Enum

class DebugStep(Enum):
    """Enum for different processing steps that can be debugged"""
    GRAYSCALE = "grayscale"
    BILATERAL = "bilateral"
    THRESHOLD = "threshold"
    MORPHOLOGY = "morphology"
    EDGES = "edges"
    FINAL = "final"
    LINES = "lines"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    COMBINED = "combined"
    CLEANED = "cleaned"
    PATTERN = "pattern"
    DENSITY = "density"
    VALID_REGIONS = "valid_regions"

class NetDetector:
    def __init__(self):
        # Initialize parameters
        self.angle_threshold = 10
        self.min_line_length_ratio = 1/3
        self.height_range = (1/4, 3/4)
        
        # Debug storage
        self.debug_frames = {}

    def detect(self, frame, debug_step=None):
        """
        Detect volleyball net in the frame focusing on the thick upper edge.
        
        Args:
            frame: Input image frame
            debug_step: DebugStep enum indicating which intermediate frame to return
            
        Returns:
            If debug_step is None: numpy.ndarray of filtered lines or None
            If debug_step is specified: corresponding debug frame
        """
        # ------------------------
        # 1. Initial Conversion
        # ------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.debug_frames[DebugStep.GRAYSCALE] = gray
        
        if debug_step == DebugStep.GRAYSCALE:
            return gray
        
        # ------------------------
        # 2. Preprocessing
        # ------------------------
        bilateral = cv2.bilateralFilter(
            gray,
            d=15,
            sigmaColor=200,
            sigmaSpace=200
        )
        # bilateral = gray
        self.debug_frames[DebugStep.BILATERAL] = bilateral
        
        if debug_step == DebugStep.BILATERAL:
            return bilateral
        
        thresh = cv2.adaptiveThreshold(
            bilateral,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=5
        )
        self.debug_frames[DebugStep.THRESHOLD] = thresh
        
        if debug_step == DebugStep.THRESHOLD:
            return thresh
        
        # ------------------------
        # 3. Morphological Operations
        # ------------------------
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (15, 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 15)
        )

        horizontal_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        vertical_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

        combined = cv2.addWeighted(horizontal_img, 0.5, vertical_img, 0.5, 0)

        self.debug_frames[DebugStep.MORPHOLOGY] = combined
        if debug_step == DebugStep.MORPHOLOGY:
            return combined

        _, pattern_mask = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
 # 2. Analyze pattern density
        kernel_size = (50, 50)  # Adjust based on your image size
        density_map = cv2.boxFilter(pattern_mask, -1, kernel_size)
        
        # Create density threshold (adjust these values based on testing)
        min_density = 30  # Minimum white pixels expected in kernel area
        max_density = 200  # Maximum white pixels expected in kernel area
        
        # 3. Filter based on pattern density
        valid_regions = np.zeros_like(pattern_mask)
        valid_regions[(density_map >= min_density) & (density_map <= max_density)] = 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(valid_regions, connectivity=8)

        # Skip label 0 as it's the background
        if num_labels > 1:
            # Get areas of all components (excluding background)
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            # Find the largest component
            largest_label = np.argmax(areas) + 1  # Add 1 because we skipped background
            
            # Create mask with only the largest component
            valid_regions = np.zeros_like(valid_regions)
            valid_regions[labels == largest_label] = 255
        
        # Store debug frames
        self.debug_frames[DebugStep.PATTERN] = pattern_mask
        self.debug_frames[DebugStep.DENSITY] = density_map
        self.debug_frames[DebugStep.VALID_REGIONS] = valid_regions
        
        if debug_step:
            return self.debug_frames.get(debug_step, frame)
        
        return valid_regions
        # dilated = cv2.dilate(thresh, horizontal_kernel, iterations=2)
        # eroded = cv2.erode(dilated, horizontal_kernel, iterations=1)
        # self.debug_frames[DebugStep.MORPHOLOGY] = eroded
        
        # if debug_step == DebugStep.MORPHOLOGY:
        #     return eroded
        
        # ------------------------
        # 4. Edge Detection
        # ------------------------
        edges = cv2.Canny(
            combined,
            threshold1=50,
            threshold2=150
        )
        self.debug_frames[DebugStep.EDGES] = edges
        
        if debug_step == DebugStep.EDGES:
            return edges
        
        # ------------------------
        # 5. Line Detection & Filtering
        # ------------------------
        # lines = cv2.HoughLinesP(
        #     edges,
        #     rho=1,
        #     theta=np.pi/180,
        #     threshold=50,
        #     minLineLength=100,
        #     maxLineGap=20
        # )
        
        # if lines is not None:
        #     # Separate vertical (poles) and horizontal (net) lines
        #     vertical_lines = []
        #     horizontal_lines = []
            
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
        #         # Vertical lines (poles) will be close to 90 degrees
        #         if 80 <= angle <= 100:
        #             vertical_lines.append(line[0])
        #         # Horizontal lines (net) will be close to 0 or 180 degrees
        #         elif angle <= 10 or angle >= 170:
        #             horizontal_lines.append(line[0])
        
        #     # Draw debug visualization
        #     debug_frame = frame.copy()
            
        #     # Draw poles in red
        #     for line in vertical_lines:
        #         x1, y1, x2, y2 = line
        #         cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        #     # Draw net in green
        #     for line in horizontal_lines:
        #         x1, y1, x2, y2 = line
        #         cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        #     self.debug_frames[DebugStep.LINES] = debug_frame
            
        #     # Find intersection points between poles and net
        #     intersections = []
        #     for v_line in vertical_lines:
        #         for h_line in horizontal_lines:
        #             # Calculate intersection point
        #             # ... (we can add intersection calculation if needed)
        #             pass
        
        self.debug_frames[DebugStep.FINAL] = debug_frame
        
        if debug_step == DebugStep.FINAL:
            return debug_frame
            
        return np.array(horizontal_lines) if horizontal_lines else None

    def get_debug_frame(self, step):
        """Get a specific debug frame"""
        return self.debug_frames.get(step) 
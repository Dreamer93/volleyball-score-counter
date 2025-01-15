from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import importlib
import time
import sys
import cv2
import os
import volleyball_detector
import net_detector

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = time.time()
        self.window_name = "Volleyball Court Detection"
        self.running = True
        
    def on_modified(self, event):
        if (event.src_path.endswith('volleyball_detector.py') or 
            event.src_path.endswith('net_detector.py')):
            current_time = time.time()
            if current_time - self.last_modified > 1:
                print(f"\nCode changed in {event.src_path}, reloading...")
                self.last_modified = current_time
                self.reload_detector()

    def reload_detector(self):
        try:
            # First reload net_detector since volleyball_detector depends on it
            importlib.reload(net_detector)
            # Then reload volleyball_detector
            importlib.reload(volleyball_detector)
            
            # Store the current image if we have one
            current_image = None
            if hasattr(self, 'detector') and hasattr(self.detector, 'current_frame'):
                current_image = self.detector.current_frame
            
            # Create new instance
            self.detector = volleyball_detector.VolleyballScoreDetector()
            
            # Restore the image if we had one
            if current_image is not None:
                self.detector.current_frame = current_image
                
            print("Detectors reloaded successfully!")
        except Exception as e:
            print(f"Error reloading module: {str(e)}")
            import traceback
            traceback.print_exc()

    def process_and_display(self):
        try:
            # Load and process image
            image_path = "test_image2.jpeg"
            image = self.detector.load_image(image_path)
            result = self.detector.process_frame(image)
            
            # Update display
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, result)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()

def start_watching():
    handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(handler, path='.', recursive=False)
    observer.start()

    # Initial load
    handler.reload_detector()

    try:
        while handler.running:
            # Process and display current frame
            handler.process_and_display()
            
            # Check for quit
            key = cv2.waitKey(100) & 0xFF  # Reduced wait time for more responsive updates
            if key == ord('q'):
                handler.running = False
                break
                
    except KeyboardInterrupt:
        handler.running = False
        
    finally:
        observer.stop()
        cv2.destroyAllWindows()
        observer.join()

if __name__ == "__main__":
    # Install required package if not present
    try:
        import watchdog
    except ImportError:
        print("Installing watchdog...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
        import watchdog
    
    start_watching() 
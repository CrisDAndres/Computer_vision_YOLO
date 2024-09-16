import cv2

class Coordinates:
    """
    Class to manually define a polygon within the capture.
    """
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Unable to access the video.")
            return
        
        cv2.namedWindow("Frame") # Create an OpenCV window called ‘Frame’.
        cv2.setMouseCallback("Frame", self.print_coordinates) # Each time the user clicks on the window, the function print_coordinates is executed, which prints the coordinates of the click.

        self.video()

    def print_coordinates(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN: # Detects if the user has clicked the left mouse button.
            print(f"[{x}, {y}],") # Print coordinates

    def video(self):
        while True:
            status, frame = self.cap.read()
            
            if not status:
                break

            cv2.imshow("Frame", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'): break # press q to exit

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    video_path = 'data/cars.mp4'
    c = Coordinates(video_path)
    
# run in the terminal: python -B coordinates.py 

# code source: https://www.youtube.com/watch?v=sy8uRDZw8pk&t=20s
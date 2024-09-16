import cv2
import torch
import numpy as np
import matplotlib.path as mplPath

"""
CODE STEPS
1. Upload and display the video
2. Load Yolov5n model
3. Filter and get bboxes
4. Validate detections
    4.1. Define zone (polygon)
    4.2. Validate if the detection is in polygon
    4.3. Add counter 
5. Display
"""

ZONE = np.array([ # copy the coordinates obtained from the coordinates.py script
    [202, 201],
    [503, 186],
    [514, 307],
    [234, 299]
])

def get_center(bbox):
    """
    To know if the class it detects is inside the polygon we have defined,
    we have to find a reference point inside the bbox that will be used to validate
    if the bbox is inside the polygon.
    """
    # xmin, ymin, xmax, ymax
    # 0     1     2     3
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center
    
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained = True) # yolov5 nano
    return model


def get_bboxes(pred: object):
    """
    Extract bounding boxes for cars and buses.
    """
    df = pred.pandas().xyxy[0] # transform the predictions to df of pandas. xyxy are the coordinates of the bounding boxes, in xmin, ymin, xmax, ymax format.
    df = df[df["confidence"] >= 0.5] # filter by confidence
    df = df[df["name"].isin(["car", "bus"])] # filter by clases
    
    return df[["xmin", "ymin", "xmax", "ymax", "name", "confidence"]].values # Return xmin, ymin, xmax, ymax, and the class name

def is_valid_detection(xc, yc):
    """
    Validates whether the centre point is at the centre of the polygon, returning a boolean
    """
    return mplPath.Path(ZONE).contains_point((xc, yc))


def detector(cap: object):
    model = load_model()
        
    if not cap.isOpened():
        print("Unable to access the video.")
        return
    
    while True:
        status, frame = cap.read() # Read the video frame
        
        if not status:
            print("Error capturing the frame.")
            break
        
        # Inference
        pred = model(frame) # returns an object that I can manipulate as a numpy array or a pandas df
        
        bboxes = get_bboxes(pred)

        cars_count = 0
        buses_count = 0
        
        for box in bboxes:
            xmin, ymin, xmax, ymax, obj_name, confidence = box
            # Convert the bounding box coordinates to integers
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            
            xc, yc = get_center(box) # obtain the coordinates of the centre
            # Convert the center coordinates to integers
            xc, yc = int(xc), int(yc)
            
            if is_valid_detection(xc, yc):
                if obj_name == "car": 
                    cars_count += 1 # if the centre point is inside the polygon, we increase the counter
                elif obj_name == "bus":
                    buses_count += 1 # if the centre point is inside the polygon, we increase the counter
            
            # mark the centre
            cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0,255,0), thickness=1) 
            # mark the bbox
            cv2.rectangle(img=frame, pt1 = (xmin, ymin), pt2 = (xmax, ymax), color=(255, 0, 0), thickness=1) # 255, 0, 0 is the color (blue) / 1 is the thickness of the line
            # show class & score
            cv2.putText(frame, 
                    f"{obj_name} {confidence:.2f}", 
                    (xmin, ymin - 10), # localisation
                    cv2.FONT_HERSHEY_PLAIN,
                    1, # fontScale
                    (0,255,0), # color: green
                    2) # thickness 
            
            
        # show the counter
        cv2.putText(img=frame, text=f"Cars: {cars_count}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=3)   # color: white
        cv2.putText(img=frame, text=f"Buses: {buses_count}", org=(50, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=3)   # color: white
        # mark the defined polygon
        cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(0,0,255), thickness=4) # color: red | thickness: 4

          
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): # Exit the loop if you press ‘q’.
            break
    
    # Release the video and close the windows
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    video_path = 'cars.mp4'
    cap = cv2.VideoCapture(video_path) 
    detector(cap)
    

# run in the terminal: python -B detect.py
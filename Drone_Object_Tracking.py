import cv2

TrDict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'mil': cv2.TrackerMIL_create}

tracker = None
tracking = False
bbox = None

cap = cv2.VideoCapture(0)

def draw_rectangle(event, x, y, flags, param):
    global bbox, tracking, tracker
    
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x, y, 0, 0)
        tracking = False
    elif event == cv2.EVENT_LBUTTONUP:
        bbox = (bbox[0], bbox[1], x - bbox[0], y - bbox[1])
        if bbox[2] > 0 and bbox[3] > 0:  # Check if a valid bounding box was selected
            tracker = TrDict['csrt']()
            tracker.init(frame, bbox)
            tracking = True

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_rectangle)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if not ret:
        break
    
    if not tracking:
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            (x, y, w, h) = [int(a) for a in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
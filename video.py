## Original source from here:
## https://realpython.com/face-detection-in-python-using-a-webcam/

import cv2
import time

video_capture = cv2.VideoCapture(0)
folder = "./test/luca"
# detect faces in the image

# draw an image with detected objects
def draw_image_with_boxes(frame, result_list):
	
	# plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # draw the box
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        print("Exiting...")
        break
    if key  == ord('s'):
        timestr = time.strftime("%Y-%m-%d %H:%M:%S")
        print("Saving image: "+timestr + ".jpeg")
        filename = folder + timestr + ".jpeg"
        cv2.imwrite(filename, frame)
    if key == 13: ## enter key
        timestr = time.strftime("%Y-%m-%d %H:%M:%S")
        print("Saving image: "+timestr + ".jpeg")
        filename = folder + timestr + ".jpeg"
        cv2.imwrite(filename, frame)

video_capture.release()
cv2.destroyAllWindows()
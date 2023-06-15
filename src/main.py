import cv2
import numpy as np
from datetime import datetime
import logging
from logging.config import fileConfig
import time

fileConfig('logging_config.ini')
log = logging.getLogger()

WINDOW_NAME = 'Foscam Feed'
CAPTURE_ALL_TRANSFORMATION = False

# cap = cv2.VideoCapture("example2.mp4")
cap = cv2.VideoCapture()
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

def draw_text(img,
              text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


"""
Capture the raw footage from the camera.
"""
def capture_raw_video(img):
    pass


"""
Capture the dialated footage from the camera.
"""
def capture_dialated_video(img):
    pass

"""
Ignore everything within a rectangular grid whose area
is specified by the user. This is suposed to work like
cv2.rectangle. Give it a tuple start and end point and
ignore everything inside of it.

show_ignored - Visually show the space being ignored.
"""
def ignore_spot(img, start_point, end_point, show_ignored=False):
    start_x = start_point[0]
    start_y = start_point[1]
    end_x = end_point[0]
    end_y = end_point[1]

    if show_ignored:
        cv2.rectangle(frame1, start_point, end_point, (255, 0, 195), 2)

def in_ignore_spot(start_point, end_point, ignore_start, ignore_end):
    start_x = start_point[0]
    start_y = start_point[1]
    end_x = end_point[0]
    end_y = end_point[1]
    ignore_start_x = ignore_start[0]
    ignore_start_y = ignore_start[1]
    ignore_end_x = ignore_end[0]
    ignore_end_y = ignore_end[1]

    if start_x >= ignore_start_x and start_y >= ignore_start_y:
        if end_x <= ignore_end_x and end_y <= ignore_end_y:
            # log.debug('In ignore area.')
            return True
    return False


"""
Capture footage of all the transformations..
"""
def capture_all_intermediate_transformations_video(img,
    capture_all_transformation=CAPTURE_ALL_TRANSFORMATION):
    if CAPTURE_ALL_TRANSFORMATION:
        capture_raw_video(img)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

log.debug(f'Frame Width: {frame_width}')
log.debug(f'Frame Height: {frame_height}')

# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

lastDateTime = None

try:
    while cap.isOpened():
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        datetimeNow = str(datetime.now()).split(".")[0]

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 700:
                continue
            in_ignore_area = in_ignore_spot((x, y), (x + w, y + h), (460, 390), (640, 480))

            if not in_ignore_area:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 195), 2)
                cv2.putText(frame1, "Status: Movement", (175, 480), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 255, 255), 2)

            # Don't make the logs noisy with redundant records of movement.
            if lastDateTime is None or lastDateTime != datetimeNow:
                if not in_ignore_area:
                    log.info("Movement at {}".format(datetimeNow))
                    lastDateTime = datetimeNow

        cv2.putText(frame1, fps, (10, 470), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 0), 2, cv2.LINE_AA)

        ignore_spot(frame1,
                    start_point=(460, 390),
                    end_point=(640, 480),
                    show_ignored=False)

        draw_text(img=frame1,
                  text=datetimeNow,
                  pos=(10, 30),
                  font=cv2.FONT_HERSHEY_PLAIN,
                  font_thickness=1,
                  text_color=(255, 255, 255),
                  text_color_bg=(0, 0, 0))
        # cv2.putText(frame1,
        #             datetimeNow,
        #             (10, 30),
        #             (255, 255, 255),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1,
        #             (255, 255, 255), 1)

        image = cv2.resize(frame1, (1280, 720))

        # cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 195), 2)
        # cv2.imwrite("gray.jpg", gray)
        # out.write(image)
        # cv2.imshow("FoscamFeed", frame1)
        out.write(image)
        cv2.imshow(WINDOW_NAME, frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break
except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
cap.release()
out.release()

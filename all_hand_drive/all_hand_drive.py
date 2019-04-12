import cv2
from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path


VIEW_RESIZE_FACTOR = 0.5


def get_body_part(human, part):
    try:
        return human.body_parts[part.value]
    except KeyError:
        return None


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

view_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * VIEW_RESIZE_FACTOR)

width = 368
height = 368

e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(width, height))

l_power = 0
r_power = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=VIEW_RESIZE_FACTOR, fy=VIEW_RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

    humans = e.inference(frame, resize_to_default=True, upsample_size=2.5)
    image = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

    if len(humans) > 0:
        h = humans[0]

        if get_body_part(h, CocoPart.LWrist):
            l_power = 100 - (get_body_part(h, CocoPart.LWrist).y * 100)
        else:
            l_power -= 1

        if get_body_part(h, CocoPart.RWrist):
            r_power = 100 - (get_body_part(h, CocoPart.RWrist).y * 100)
        else:
            r_power -= 1

        cv2.putText(frame, 'R: %.2f | L: %.2f' % (r_power, l_power), (width//2 - 10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255))
        cv2.arrowedLine(frame, (10, height), (10, height - int(r_power * (height/100))), (0, 0, 0), thickness=2)
        cv2.arrowedLine(frame, (view_width - 10, height), (view_width - 10, height - int(l_power * (height/100))), (0, 0, 0), thickness=2)

    cv2.imshow('Input', frame)

    # Check for ESC
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

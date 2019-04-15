import cv2
import json
import paho.mqtt.client as mqtt
from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from sys import argv


VIEW_RESIZE_FACTOR = 0.5


def get_body_part(human, part):
    try:
        return human.body_parts[part.value]
    except KeyError:
        return None


def on_connect(client, userdata, flags, rc):
    print("Connected with result code ", str(rc))
    # client.subscribe('car/move')


def on_message(client, userdata, msg):
    print(msg.topic + ": " + str(msg.payload))


def on_log(client, userdata, level, buf):
    print("log: ", buf)


client = None

if len(argv == 5):
    host = argv[1]
    port = argv[2]
    username = argv[3]
    password = argv[4]

    client = mqtt.Client()
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_log = on_log

    client.username_pw_set(username, password)
    client.connect(host, int(port))

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

        if client is not None:
            client.publish('car/move', json.dumps({'l': l_power, 'r': r_power, 'd': 1}))

    cv2.imshow('Input', frame)

    # Check for ESC
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

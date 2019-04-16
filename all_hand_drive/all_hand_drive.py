import cv2
from datetime import datetime
import json
import paho.mqtt.client as mqtt
from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from sys import argv


COUNTDOWN_TIME = 3
VIEW_RESIZE_FACTOR = 0.5


def get_body_part(human, part):
    try:
        return human.body_parts[part.value]
    except KeyError:
        return None


def get_part_coordinates(part, x_offset=0, y_offset=0):
    return int(part.x * view_width + x_offset), int(part.y * view_height + y_offset)


def wrist_to_hand_coordinates(part):
    return get_part_coordinates(part, y_offset=-20)


def get_body_coordinates(human, part):
    part = get_body_part(human, part)

    if part:
        return int(part.x * view_width), int(part.y * view_height)


def draw_steering_wheel(frame):
    cv2.circle(frame, wheel_center, wheel_radius, (0, 0, 0), 8)


def calculate_power(part):
    power = 100 - (wrist_to_hand_coordinates(part)[1] - (wheel_center[1] - wheel_radius//2))

    if power > 100:
        return 100
    elif power < 0:
        return 0
    else:
        return power


def decay_power(power):
    power -= 1

    if power < 0:
        return 0
    else:
        return power


def on_connect(client, userdata, flags, rc):
    print("Connected with result code ", str(rc))
    # client.subscribe('car/move')


def on_message(client, userdata, msg):
    print(msg.topic + ": " + str(msg.payload))


def on_log(client, userdata, level, buf):
    print("log: ", buf)


client = None

if len(argv) == 5:
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
view_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * VIEW_RESIZE_FACTOR)

width = 368
height = 368

e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(width, height))

l_power = 0
r_power = 0

wheel_center = 0
wheel_radius = 0

start_countdown = datetime.now()

countdown = False
driving = False
elapsed = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=VIEW_RESIZE_FACTOR, fy=VIEW_RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

    humans = e.inference(frame, resize_to_default=True, upsample_size=2.5)
    # image = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

    if len(humans) > 0:
        h = humans[0]

        l_wrist = get_body_part(h, CocoPart.LWrist)
        r_wrist = get_body_part(h, CocoPart.RWrist)

        hand_colour = (0, 0, 255)

        if l_wrist and r_wrist:
            hand_colour = (0, 255, 0)

        if l_wrist:
            cv2.circle(frame, wrist_to_hand_coordinates(l_wrist), 20, hand_colour, 3)

        if r_wrist:
            cv2.circle(frame, wrist_to_hand_coordinates(r_wrist), 20, hand_colour, 3)

        if not driving:
            if countdown:
                elapsed = int((datetime.now() - start_countdown).total_seconds())
                count = COUNTDOWN_TIME - elapsed

                light_colour = (0, 0, 255)

                if count <= 0:
                    light_colour = (0, 255, 0)
                elif count <= 1:
                    light_colour = (0, 165, 255)

                nose = get_body_coordinates(h, CocoPart.Nose)

                cv2.circle(frame, (nose[0], nose[1] - 100), 20, light_colour, -1)
                if count > 0:
                    cv2.putText(frame, '%.2f' % (COUNTDOWN_TIME - (datetime.now() - start_countdown).total_seconds()), (wheel_center[0] - 50, wheel_center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness=2)

            if l_wrist and r_wrist:
                l_coordinates = wrist_to_hand_coordinates(l_wrist)
                r_coordinates = wrist_to_hand_coordinates(r_wrist)

                wheel_center = ((l_coordinates[0] + r_coordinates[0])//2, (l_coordinates[1] + r_coordinates[1])//2)
                wheel_radius = wheel_center[0] - r_coordinates[0]

                draw_steering_wheel(frame)

                if not countdown:
                    start_countdown = datetime.now()
                    countdown = True
            elif countdown:
                countdown = False

            if countdown and elapsed >= COUNTDOWN_TIME:
                countdown = False
                driving = True
        else:
            draw_steering_wheel(frame)

            if l_wrist:
                cv2.circle(frame, wrist_to_hand_coordinates(l_wrist), 20, hand_colour, 3)
                l_power = calculate_power(l_wrist)
            else:
                l_power = decay_power(l_power)

            if r_wrist:
                cv2.circle(frame, wrist_to_hand_coordinates(r_wrist), 20, hand_colour, 3)
                r_power = calculate_power(r_wrist)
            else:
                r_power = decay_power(r_power)

            if not l_wrist and not r_wrist:
                l_power = 0
                r_power = 0

            cv2.putText(frame, 'R: %.2f | L: %.2f' % (r_power, l_power), (width//2 - 10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255))
            cv2.arrowedLine(frame, (10, height), (10, height - int(r_power * (height/100))), (0, 0, 0), thickness=2)
            cv2.arrowedLine(frame, (view_width - 10, height), (view_width - 10, height - int(l_power * (height/100))), (0, 0, 0), thickness=2)

            if client is not None:
                client.publish('car/drive', json.dumps({'l': l_power, 'r': r_power, 'd': 500}))
    else:
        driving = False
        countdown = False

    cv2.imshow('Input', frame)

    # Check for ESC
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

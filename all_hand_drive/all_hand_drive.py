import cv2
from datetime import datetime
import json
from messaging.messaging import MqttClient
from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from sys import argv


COUNTDOWN_TIME = 3
VIEW_RESIZE_FACTOR = 0.5


def decay_power(power):
    power -= 1

    if power < 0:
        return 0
    else:
        return power


def get_body_part(human, part):
    try:
        return human.body_parts[part.value]
    except KeyError:
        return None


class AllHandDrive:
    def wrist_to_hand_coordinates(self, part):
        return self.get_part_coordinates(part, y_offset=-20)

    def get_part_coordinates(self, part, x_offset=0, y_offset=0):
        return int(part.x * self.view_width + x_offset), int(part.y * self.view_height + y_offset)

    def get_body_coordinates(self, human, part):
        part = get_body_part(human, part)

        if part:
            return int(part.x * self.view_width), int(part.y * self.view_height)

    def draw_steering_wheel(self, frame):
        cv2.circle(frame, self.wheel_center, self.wheel_radius, (0, 0, 0), 8)

    def calculate_power(self, part):
        power = 100 - (self.wrist_to_hand_coordinates(part)[1] - (self.wheel_center[1] - self.wheel_radius//2))

        if power > 100:
            return 100
        elif power < 0:
            return 0
        else:
            return power

    def drive(self):
        command = {'l': self.l_power, 'r': self.r_power, 'd': 500}

        if self.client is not None:
            self.client.publish('car/drive', command)
        else:
            print(json.dumps(command))

    def __init__(self, host=None, port=None, username=None, password=None):
        self.client = None

        if host and port and username and password:
            self.client = MqttClient(host, port, username, password)

        self.width = 368
        self.height = 368

        self.l_power = 0
        self.r_power = 0

        self.wheel_center = 0
        self.wheel_radius = 0

        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(self.width, self.height))
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.view_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * VIEW_RESIZE_FACTOR)
        self.view_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * VIEW_RESIZE_FACTOR)

    def start(self, debug=False):
        start_countdown = datetime.now()

        countdown = False
        driving = False
        elapsed = 0

        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, None, fx=VIEW_RESIZE_FACTOR, fy=VIEW_RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

            humans = self.e.inference(frame, resize_to_default=True, upsample_size=2.5)

            if debug:
                TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

            if len(humans) > 0:
                h = humans[0]

                l_wrist = get_body_part(h, CocoPart.LWrist)
                r_wrist = get_body_part(h, CocoPart.RWrist)

                hand_colour = (0, 0, 255)

                if l_wrist and r_wrist:
                    hand_colour = (0, 255, 0)

                if l_wrist:
                    cv2.circle(frame, self.wrist_to_hand_coordinates(l_wrist), 20, hand_colour, 3)

                if r_wrist:
                    cv2.circle(frame, self.wrist_to_hand_coordinates(r_wrist), 20, hand_colour, 3)

                if not driving:
                    if countdown:
                        elapsed = int((datetime.now() - start_countdown).total_seconds())
                        count = COUNTDOWN_TIME - elapsed

                        light_colour = (0, 0, 255)

                        if count <= 0:
                            light_colour = (0, 255, 0)
                        elif count <= 1:
                            light_colour = (0, 165, 255)

                        nose = self.get_body_coordinates(h, CocoPart.Nose)

                        cv2.circle(frame, (nose[0], nose[1] - 100), 20, light_colour, -1)
                        if count > 0:
                            cv2.putText(frame, '%.2f' % (COUNTDOWN_TIME - (datetime.now() - start_countdown).total_seconds()), (self.wheel_center[0] - 50, self.wheel_center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness=2)

                    if l_wrist and r_wrist:
                        l_coordinates = self.wrist_to_hand_coordinates(l_wrist)
                        r_coordinates = self.wrist_to_hand_coordinates(r_wrist)

                        self.wheel_center = ((l_coordinates[0] + r_coordinates[0])//2, (l_coordinates[1] + r_coordinates[1])//2)
                        self.wheel_radius = self.wheel_center[0] - r_coordinates[0]

                        self.draw_steering_wheel(frame)

                        if not countdown:
                            start_countdown = datetime.now()
                            countdown = True
                    elif countdown:
                        countdown = False

                    if countdown and elapsed >= COUNTDOWN_TIME:
                        countdown = False
                        driving = True
                else:
                    self.draw_steering_wheel(frame)

                    if l_wrist:
                        cv2.circle(frame, self.wrist_to_hand_coordinates(l_wrist), 20, hand_colour, 3)
                        self.l_power = self.calculate_power(l_wrist)
                    else:
                        self.l_power = decay_power(self.l_power)

                    if r_wrist:
                        cv2.circle(frame, self.wrist_to_hand_coordinates(r_wrist), 20, hand_colour, 3)
                        self.r_power = self.calculate_power(r_wrist)
                    else:
                        self.r_power = decay_power(self.r_power)

                    if not l_wrist and not r_wrist:
                        self.l_power = 0
                        self.r_power = 0

                    cv2.putText(frame, 'R: %.2f | L: %.2f' % (self.r_power, self.l_power), (self.width//2 - 10, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255))
                    cv2.arrowedLine(frame, (10, self.height), (10, self.height - int(self.r_power * (self.height/100))), (0, 0, 0), thickness=2)
                    cv2.arrowedLine(frame, (self.view_width - 10, self.height), (self.view_width - 10, self.height - int(self.l_power * (self.height/100))), (0, 0, 0), thickness=2)

                    self.drive()
            else:
                driving = False
                countdown = False

            cv2.imshow('Input', frame)

            # Check for ESC
            c = cv2.waitKey(1)
            if c == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


all_hand_drive = None

if len(argv) == 5:
    all_hand_drive = AllHandDrive(argv[1], argv[2], argv[3], argv[4])
else:
    all_hand_drive = AllHandDrive()

all_hand_drive.start()

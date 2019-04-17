import json
import paho.mqtt.client as mqtt


class MqttClient:
    def _on_connect(self, client, userdata, flags, rc):
        print("Connected with result code ", str(rc))

    def _on_message(self, client, userdata, msg):
        print(msg.topic + ": " + str(msg.payload))

    def _on_log(self, client, userdata, level, buf):
        print("log: ", buf)

    def publish(self, topic, message):
        self.client.publish(topic, json.dumps(message))

    def __init__(self, host, port, username, password, tls=True):
        self.client = mqtt.Client()

        if tls:
            self.client.tls_set()

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_log = self._on_log

        self.client.username_pw_set(username, password)
        self.client.connect(host, int(port))

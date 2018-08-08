import click
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import tfinterface as ti
import dicto as do
import matplotlib.pyplot as plt
import seaborn as sns


TRAINING_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "config", "train.yml")
RUN_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "config", "run.yml")


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.
        self.throttle = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement, target = None):
        if target is None:
            target = self.set_point

        # proportional error
        self.error = target - measurement

        # integral error
        self.integral += self.error
        # self.integral *= 0.99

        self.throttle = self.Kp * self.error + self.Ki * self.integral

        return max(self.throttle, -0.01)




class Car:

    def __init__(self, sio, model, controller, train, params, image_folder = None):
        self.sio = sio
        self.model = model
        self.controller = controller
        self.image_folder = image_folder
        self.nbins = train.nbins
        self.params = params

        self.angles = None
        self.first_plot = True


    
    def telemetry(self, sid, data):
        if data:
            # The current steering angle of the car
            steering_angle = data["steering_angle"]
            # The current throttle of the car
            throttle = data["throttle"]
            # The current speed of the car
            speed = data["speed"]
            # The current image from the center camera of the car
            imgString = data["image"]
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_array = np.asarray(image)

            predictions = self.model.predict(
                image = [image_array]
            )

            probabilities = predictions["probabilities"][0]

            if "embedding" in predictions:
                embedding = predictions["embedding"][0]
            else:
                embedding = None

            if self.angles is None:
                self.angles = np.linspace(-1.0, 1.0, len(probabilities))
                print("Angles:", self.angles)
            

            if self.params.policy == "mean":
                steering_angle = np.dot(probabilities.T, self.angles)
            
            elif self.params.policy == "mode":
                i = np.argmax(probabilities)
                steering_angle = self.angles[i]
            elif self.params.policy == "mean_mode":
                mean = np.dot(probabilities.T, self.angles)
                i = np.argmax(probabilities)
                mode = self.angles[i]
                steering_angle = (mean + mode) / 2.0

            # target_speed = self.controller.set_point * (1.0 - ((abs(steering_angle)) / 4.0))
            # throttle = self.controller.update(float(speed), target=target_speed)
            throttle = self.controller.update(float(speed))
            
            
            # throttle = self.controller.set_point

            # print(steering_angle, throttle)
            self.send_control(steering_angle, throttle)

            # save frame
            if self.image_folder:
                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = os.path.join(self.image_folder, timestamp)
                image.save('{}.jpg'.format(image_filename))

            if self.params.plot:
                self.plot(probabilities, embedding)

        else:
            # NOTE: DON'T EDIT THIS.
            self.sio.emit('manual', data={}, skip_sid=True)

    def plot(self, probabilities, embedding):

        if self.first_plot:
            self.first_plot = False

            plt.ion()
            self.fig = plt.figure()

            label = [ "{:.3f}".format(a) for a in self.angles ]
            index = range(len(label))
            self.ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
            self.ax1.set_xticks(index)
            self.ax1.set_xticklabels(label)
            self.bar = self.ax1.bar(index, np.ones_like(probabilities))

            if embedding is not None:
                label = range(len(embedding))
                index = label
                self.ax2 = plt.subplot2grid((5, 1), (4, 0))
                self.ax2.set_xticks(index)
                self.ax2.set_xticklabels(label)
                self.bar2 = self.ax2.bar(index, np.ones_like(embedding))
            

            plt.pause(0.00001)

        else:

            for i, p in enumerate(probabilities):
                self.bar[i].set_height(p)

            if embedding is not None:
                for i, p in enumerate(embedding):
                    self.bar2[i].set_height(p)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()




    def connect(self, sid, environ):
        print("connect ", sid)
        self.send_control(0, 0)


    def send_control(self, steering_angle, throttle):
        self.sio.emit(
            "steer",
            data = {
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True
        )

    def register(self):

        self.sio.on('telemetry')(self.telemetry)
        self.sio.on('connect')(self.connect)


@click.command()
@click.option("--export-dir", required = True)
@do.click_options_config(TRAINING_PARAMS_PATH, "train", underscore_to_dash = False)
@do.click_options_config(RUN_PARAMS_PATH, "params", underscore_to_dash = False)
def main(export_dir, train, params):

    sio = socketio.Server()
    app = Flask(__name__)


    controller = SimplePIController(params.kp, params.ki)
    controller.set_desired(params.speed)

    model = ti.estimator.SavedModelPredictor(export_dir)

    car = Car(sio, model, controller, train, params)
    car.register()
    
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

if __name__ == '__main__':
    main()
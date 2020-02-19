from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import imutils
import cv2
import numpy as np

import tensorflow as tf
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# Main calss that is responsible for running GUI and making classification magic
class AiMaschroomer:
    def __init__(self, vs, outputPath, model, classes, mean = None):
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.text = None
        self.image = None
        self.thread = None
        self.stopEvent = None
        self.imgTaken = False

        self.model = model
        self.classes = classes
        self.mean = mean

        self.guiInit()
        self.mainLoopInit()
        self.text.insert(tki.END, "Make mushroom photo!")

    # Main loop - it only runs two methods
    def videoLoop(self):
        self._initPanel()

        while not self.stopEvent.is_set():
             self._showLiveImageOnPanel()

    # GUI initialization method
    def guiInit(self):
        self.root = tki.Tk()
        self.panel = None
        btn = tki.Button(self.root, text="Make a photo!", command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        self.text = tki.Text(self.root, height=1, width=30)
        self.text.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)


    def mainLoopInit(self):
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        self.root.wm_title("Maschoroom Detector")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    # Ai magic is done here!
    def takeSnapshot(self):
        print("[INFO] Checking...")
        self.text.delete("1.0", "end")
        self.imgTaken = True

        # Take a pfoto
        frame = self.frame.copy()
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame = frame.astype("float") / 255.0
        if self.mean is not None:
            (B, G, R) = cv2.split(frame)
            R = R - self.mean["R"]
            G = G - self.mean["G"]
            B = B - self.mean["B"]
            frame = cv2.merge([B, G, R])

        frame = np.expand_dims(frame, axis=0)

        #Pass it to MobileNet model
        label = self.model.predict(frame)
        label = np.round(label, 2)*100
        label = label.astype(int)
        max = np.max(label[0])
        print(max)
        idx = np.where(label[0]==max)

        # Print label
        self.text.insert(tki.END, self.classes[idx[0][0]])
        print("[INFO] Label: ", self.classes[idx[0][0]], label)

    def onClose(self):
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()

    def _initPanel(self):
        self._getImage()
        # if the panel is not None, we need to initialize it
        if self.panel is None:
            self.panel = tki.Label(image=self.image)
            self.panel.image = self.image
            self.panel.pack(side="left", padx=10, pady=10)

    def _getImage(self):
        self.frame = self.vs.read()
        self.frame = imutils.resize(self.frame, width=400, height=400)

        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image = image

    def _showImage(self):
        self.panel.configure(image=self.image)
        self.panel.image = self.image

    def _showStaticImageOnPanel(self):
        self._showImage()

    def _showLiveImageOnPanel(self):
        self._getImage()
        self._showImage()


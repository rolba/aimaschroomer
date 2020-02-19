from __future__ import print_function
from aimaschroomer import AiMaschroomer
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
import argparse
import time
import os
import json
# Main application prepared for Raspbery pi.


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

splittedSetPath = os.getcwd() + "/data_h5py"

# Load serialized model
modelsPath = os.path.join(splittedSetPath, "models")
modelPath = os.path.join(modelsPath, "mobilenetv2.model")
model = load_model(modelPath)

# Load mean of training data
meansPath = os.path.join(splittedSetPath, "mean", "train_mean.json")
means = json.loads(open(meansPath).read())

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# Let's roll...
pba = AiMaschroomer(vs, "output", model, ["Eatable", "Poisoned", "Uneatable"], means)
pba.root.mainloop()
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json

# Training monitor was needed for testing training process. I like to see whe model overfits.

class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None):

		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath

	def on_train_begin(self, logs={}):
		# initialize the history dictionary
		self.H = {}


	def on_epoch_end(self, epoch, logs={}):
		# loop over the logs and update the loss, accuracy, etc.
		# for the entire training process
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l

		# check to see if the training history should be serialized
		# to file
		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H))
			f.close()

		# ensure at least two epochs have passed before plotting
		# (epoch starts at zero)
		if len(self.H["loss"]) > 1:
			# plot the training loss and accuracy
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			plt.plot(N, self.H["val_loss"], label="val_loss")
			plt.plot(N, self.H["accuracy"], label="train_accuracy")
			plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
			plt.title("Training Loss and Accuracy [Epoch {}]".format(
				len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()

			# save the figure
			plt.savefig(self.figPath)
			plt.close()
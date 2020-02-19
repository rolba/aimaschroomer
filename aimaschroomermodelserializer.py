import os
from tensorflow.keras.optimizers import Adam, SGD
import mobilenet

# I need a model serializer. So I wrote one.

def main():
    # load weights
    modelsPath = os.getcwd() + "/data_h5py/models"
    modelPath = os.path.join(modelsPath, "weights-049-0.0160_mobilenetv2.hdf5")

    # Construct model
    model = mobilenet.MobileNetv2((224, 224, 3), 3, 1.0)
    opt = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.load_weights(modelPath)

    # Serialize model to HDD
    modelsPath = os.getcwd() + "/data_h5py/models"
    modelPath = os.path.join(modelsPath, "mobilenetv2.model")
    model.save(modelPath, overwrite=True)

if __name__ == "__main__":
    main()
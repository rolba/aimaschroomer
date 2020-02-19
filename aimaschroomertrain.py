from DataGenerator import DataGenerator as dg
import os
import h5py
import json
import trainingmonitor
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import mobilenet

def main():
    splittedSetPath = os.getcwd() + "/data_h5py"
    trainingSetPath = os.path.join(splittedSetPath, "trainset", "trainSet.h5py")
    testingSetPath = os.path.join(splittedSetPath, "testset", "testSet.h5py")
    validatingPath = os.path.join(splittedSetPath, "validationset", "validateSet.h5py")
    labelsPath = os.path.join(splittedSetPath, "labels", "labels.json")
    meansPath = os.path.join(splittedSetPath, "mean", "train_mean.json")

    # Load train set mean value.
    means = json.loads(open(meansPath).read())
    print(means)

    # Load hdf5 data sets of train, test and validate data
    trainDb = h5py.File(trainingSetPath, "r")
    print(trainDb["labels"].shape[0])
    testDb = h5py.File(testingSetPath, "r")
    print(testDb["labels"].shape[0])
    valDb = h5py.File(validatingPath, "r")
    print(valDb["labels"].shape[0])

    # Prepare MobileNetV2 model.
    model = mobilenet.MobileNetv2((224, 224, 3), 3, 1.0)
    # Prepare optimize object. I will use Adam with different lerning rates. It's Hyper parameter here.
    opt = Adam(lr=1e-3)
    # Now let's compile AlexNet to be digestable by hardware (GPU). This methood prepares model for training.
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    trainGenerator = dg(trainDb, batchSize=32, aug=None, binarize=True, classesNum=3)
    trainGenerator.setimageresizer(width=224, height=224)
    trainGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])

    validateGenerator = dg(valDb, batchSize=32, aug=None, binarize=True, classesNum=3)
    validateGenerator.setimageresizer(width=224, height=224)
    validateGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])

    modelsPath = os.path.join(splittedSetPath, "models")
    fname = os.path.sep.join([modelsPath, "weights-{epoch:03d}-{val_loss:.4f}_mobilenetv2.hdf5"])
    checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only = True, verbose = 1)

    monitorPath = os.path.join(splittedSetPath, "monitor", "monitor_mobilenetv2.png")
    trm = trainingmonitor.TrainingMonitor(monitorPath)

    callbacks = [checkpoint, trm ]

    # Train the network using generated data.
    H = model.fit(
        trainGenerator.generator(),
        steps_per_epoch=trainGenerator.numImages // 32,
        validation_data=validateGenerator.generator(),
        validation_steps=validateGenerator.numImages // 32,
        epochs=50,
        max_queue_size=32 * 2,
        verbose=1,
        workers = 1,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
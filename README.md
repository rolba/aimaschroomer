# aimaschroomer

Training, data preparation, scraping scripts:
1. I created https://github.com/rolba/rScraper API for downloading data from the internet for training purposes. Created for this project.
2. aimaschroomerprepare.py - File that generates sugumented data for training purposes.
3. aimaschroomersplitter.py - Splits data into train set, validate set and test set.
4. aimaschroomerhdf5generator.py - Script that generates HDF5 file containers for thraining purposes.
5. aimaschroomermodelserializer.py - Converts saved weights to serialized mode.
6. aimaschroomertrain.py - Trains MobileNet deep neural network and uses data generated in HDF5 containers.

Helpers:
1. trainingmonitor.py - Helps to take care of training process - I can easily find out that I overfit by printing plots per every epoch.
2. DataGenerator.py - helper class for preprocessing images before they are send to hdf5 data file containers. (Reused)
3. DatasetWriter.py - helper class that puts images into hdf5 data file container. (Reused)
4. mobilenet.py - Taken from Keras Repo.

Main App.
1. main.py - Main GUI application file. Runs propoer classes.
2. mobilenet.py - Logic class used for creating GUI, and all logic that is needed for taking a photo, passing it to model and make classification if muschroom is Eatable or poisoned.
    

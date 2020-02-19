# import the necessary packages
import cv2
from os.path import join
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Overall idea is to create a script that can help genrate Augumented images.

os.environ["PATH"] += os.pathsep + os.getcwd()
datasetPath = os.getcwd() + "/data"
datasetPreparedPath = os.getcwd() + "/data_prep"
labels = ["Eatable", "Poisoned", "Uneatable"]

def generate(img, saveDir, prefix):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range = [0.8, 1.2],
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest')

    x = img_to_array(img)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0

    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    for batch in datagen.flow(x, batch_size=1, save_to_dir=saveDir, save_prefix=prefix, save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely

# Generally I am iterating over downloaded images and preparing augumented data
def main():
    for label in labels:
        i = 0
        labelPath = os.path.join(datasetPath, label)
        dirsList = os.listdir(labelPath)
        print(label)
        for specious in dirsList:
            print(specious)
            imgsDir = os.path.join(labelPath, specious, "ok")
            imgsFnames = os.listdir(imgsDir)
            for imgFileName  in imgsFnames:
                imgDir = os.path.join(imgsDir, imgFileName)

                frame = load_img(imgDir)
                if frame is not None:
                    generate(frame, join(datasetPreparedPath, label), str(specious)+"_"+str(i))
                    i = i+1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

model=VGG16()
image1 = load_img('C:/Users/zaxna/Desktop/M1-STIC/APM2/reaper.jpeg', target_size=(224, 224))
image2 = load_img('C:/Users/zaxna/Desktop/M1-STIC/APM2/loup.jpeg', target_size=(224, 224))
image3 = load_img('C:/Users/zaxna/Desktop/M1-STIC/APM2/ballon.jpeg', target_size=(224, 224))


def preprocess(image):
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)

    return image

def pred_modele(image):

    image = preprocess(image)
    y_pred = model.predict(image)
    label=decode_predictions(y_pred)
    label=label[0][0]
    return ((label[1], label[2]*100))

img =[image1, image2, image3]

for i in range(3):
   print("prediction for image", i+1, ":", pred_modele(img[i])[0], "Avec une proablilité de ",round(pred_modele(img[i])[1], 2), "%")
# pretrained model taken from https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import load_model
import numpy as np
import cv2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image


model = MobileNetV2(weights='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')

#img_path = 'lynx.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#y = model.predict(x)
#decoded = decode_predictions(y, top=3)[0]
#labels = np.array(decoded)[:,1:3]
#print(labels)

#a = 1/0

cap = cv2.VideoCapture(1)

while True:
    # read image
    ret, im_src = cap.read()

    # crop
    height, length = im_src.shape[0], im_src.shape[1]
    size = min(height, length)
    im_cropped = im_src[height//2 - size//2:height//2 + size//2, length//2 - size//2:length//2 + size//2]

    # scale
    im_scaled = cv2.resize(im_cropped, (224,224), interpolation = cv2.INTER_CUBIC)

    # predict
    x = image.img_to_array(im_scaled)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    decoded = decode_predictions(y, top=3)[0]
    labels = np.array(decoded)[:,1:3]
    print(labels)

    # create label
    p = 0
    im_label = np.zeros((224,224,3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im_label, str(p),(224//2,224//2), font, 8, (255,255,255), 10, cv2.LINE_AA)

    # show images
    cv2.imshow('webcam', im_src)
    #cv2.imshow('cropped', im_cropped)
    cv2.imshow('scaled', im_scaled)
    cv2.imshow('label', im_label)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

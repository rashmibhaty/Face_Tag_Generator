# Face_Tag_Generator
Machine learning techniques are used to automatically tag Images and Videos.

Generating the model:
Face extracion:
File:face_extract.py
Face detection technique is used to extract gray scale face images. Opencv CascadeClassifier detectMultiScale are the the api used for detection.

Training the model:
File:face_model.py
Extracted faces are devided into training and validation set.
CNN model is trained using the faces. 
ImageDataGenerator and flow_from_directory are used for get the images.
Loss is set to "categorical_crossentropy" 
Trained model is saved. 

Automatic Tagging:
File:face_tag.py
     face_tag_video.py
Trined model and indices are loaded. load_model is the apis used.
Faces detected in new inamges and videos are used for prediction using the saved model.model.predict is the api used.

     

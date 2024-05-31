import os
import face_recognition
import pickle


face_dir = 'face_images/'

##Initailizing data structure
known_face_encodings = []
known_face_names = []

#Loop through images in directory
for filename in os.listdir(face_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.gif') or filename.endswith('.bmp') or filename.endswith('.tif') or filename.endswith('.tiff'):
        image_path = os.path.join(face_dir, filename)
        image = face_recognition.load_image_file(image_path)

        #Get face encoding
        face_encoding = face_recognition.face_encodings(image)[0]

        # Adding the encoding and the name (from the filename)
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

#Saving the encoding and names to a file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Face encodings saved to face_encodings.pkl")


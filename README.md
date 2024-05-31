**Facial Recognition Scanner**
Overview
The Facial Recognition Scanner project is designed to detect and recognize faces in images using the face_recognition and opencv-python libraries. This project includes a backend script to encode faces and a 
frontend application to utilize these encodings for facial recognition tasks.

**Features**
Detect and recognize faces in images.
Encode face images and save the encodings.
Compare faces to determine matches.


**Requirements**
Python 3.6+
face_recognition library
opencv-python library


**Installation**
Clone the repository:
Copy code
git clone https://github.com/yourusername/Facial_Recognition_Scanner.git
cd Facial_Recognition_Scanner/face-scanner-backend
Create and activate a virtual environment (optional but recommended):


**Before running the main application, you need to generate face encodings. Place your images in the images directory and run:**
Copy code
python generate_face_encodings.py
This will create a face_encodings.pkl file with the encoded faces.

**Run the Facial Recognition Script:**
Use the app.py script to perform facial recognition:
python app.py

**Acknowledgments**
face_recognition library by Adam Geitgey
OpenCV library

**Contact**
For any questions or issues, please contact [awale905@gmail.com] or open an issue on GitHub.

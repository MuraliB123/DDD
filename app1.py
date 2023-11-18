from flask import Flask, render_template, Response
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import time

app = Flask(__name__)


model = load_model("drowsiness_new6(1).h5")


img_size = (145, 145)

def preprocess_image(img):
    img_array = cv2.resize(img, img_size)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array

def observe_for_duration(duration=5):
    cap = cv2.VideoCapture(0)  

    while True:
        start_time = time.time()
        observations = []

        while time.time() - start_time < duration:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Preprocess the frame and make predictions
                img_array = preprocess_image(frame)
                predictions = model.predict(img_array)

                # Assuming prediction is a 1D array of class indices
                prediction_class = np.argmax(predictions, axis=1)

                # Get the class label based on your class mapping
                class_labels = ["yawn", "no_yawn", "Closed", "Open"]
                predicted_class_label = class_labels[prediction_class[0]]

                # Append the prediction to the observations list
                observations.append(predicted_class_label)

                # Display the prediction on the frame
                cv2.putText(frame, f'Prediction: {predicted_class_label}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame for live display
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Calculate the most frequent observation in the last 3 seconds
        most_frequent_observation = max(set(observations), key=observations.count)
        
        # Print the final observation in the terminal
        print(f'Final Observation: {most_frequent_observation}')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(observe_for_duration(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

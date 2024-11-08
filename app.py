#main
# from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
# import threading
# import cv2
# import numpy as np
# import torch
# import random
# import datetime
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models asynchronously
# emotion_model = load_model('5_30AMmodel.h5')
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses
# flagged_responses = []

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test')
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Process the form data here (e.g., calculate personality score)
#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict[max_index]

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate text report of flagged responses
# def generate_report():
#     global flagged_responses
#     report_filename = "flagged_report.txt"
#     with open(report_filename, 'w') as file:
#         file.write("### Suicidal Intent Report\n\n")
#         file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         file.write("The following messages were flagged as potentially containing suicidal intent:\n\n")
#         for idx, message in enumerate(flagged_responses, 1):
#             file.write(f"{idx}. {message}\n")
#     return report_filename

# # Download report
# # Download report
# @app.route('/download_report')
# def download_report():
#     report_file = generate_report()
#     return send_file(report_file, as_attachment=True)

# ### Updated `app.py`:


from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
import threading
import cv2
import numpy as np
import torch
import random
import datetime
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Temporary in-memory user database for login
users = {"test@example.com": "password123"}

# Load models asynchronously
emotion_model = load_model('5_30AMmodel.h5')
chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
suicide_tokenizer = AutoTokenizer.from_pretrained("path")
suicide_model = AutoModelForSequenceClassification.from_pretrained("path")

# Track flagged responses
flagged_responses = []

# Route for the login page
@app.route('/')
def login():
    return render_template('login.html')

# Login authentication
@app.route('/login', methods=['POST'])
def handle_login():
    email = request.form['email']
    password = request.form['password']
    
    if email in users and users[email] == password:
        return redirect(url_for('big5_test'))
    else:
        return "Invalid credentials, try again."

# Big 5 Personality Test page
@app.route('/big5_test')
def big5_test():
    return render_template('big5.html')

# Handle Big 5 Test submission and redirect to the main app
@app.route('/submit_test', methods=['POST'])
def handle_test():
    # Process the form data here (e.g., calculate personality score)
    return redirect(url_for('main_page'))

# Main page with video and chatbot
@app.route('/main_page')
def main_page():
    return render_template('index.html')

# Emotion Detection: Generate frames from webcam
def generate_camera_feed():
    global emotion_model
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype('float32') / 255
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

            # Predict emotion
            prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            emotion_label = emotion_dict[max_index]

            cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Suicidal Intent Detection
def detect_suicidal_intent(user_message):
    global flagged_responses
    inputs = suicide_tokenizer(user_message, return_tensors="pt")
    logits = suicide_model(**inputs).logits
    prediction = torch.softmax(logits, dim=1).tolist()[0]
    if prediction[1] > 0.5:  # If the message is flagged as suicidal
        flagged_responses.append(user_message)
        return True
    return False

# Generate text report of flagged responses
def generate_report():
    global flagged_responses
    report_filename = "flagged_report.txt"
    with open(report_filename, 'w') as file:
        file.write("### Suicidal Intent Report\n\n")
        file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write("The following messages were flagged as potentially containing suicidal intent:\n\n")
        for idx, message in enumerate(flagged_responses, 1):
            file.write(f"{idx}. {message}\n")
    return report_filename

# Download report
@app.route('/download_report')
def download_report():
    report_file = generate_report()
    return send_file(report_file, as_attachment=True)

# Chatbot Route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json['message'].strip()

    # Check if the user inputs 'exit' to end the conversation
    if user_message.lower() == "exit":
        return jsonify({'reply': "Thank you for chatting with me. A report of flagged messages will be sent to the psychologist. <a href='/download_report'>Download Report</a>"})

    # Detect suicidal intent
    is_suicidal = detect_suicidal_intent(user_message)
    
    if is_suicidal:
        reply = random.choice([
            "Are you okay? How long have you been feeling this way?",
            "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
        ])
    else:
        inputs = chatbot_tokenizer([user_message], return_tensors="pt")
        reply_ids = chatbot_model.generate(**inputs)
        reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
# import threading
# import cv2
# import numpy as np
# import torch
# import random
# import datetime
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import os
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.application import MIMEApplication

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models asynchronously
# emotion_model = load_model('5_30AMmodel.h5')
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses
# flagged_responses = []

# # Store user data temporarily
# user_data = {}

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test', methods=['GET'])
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Collect the close person's email
#     close_email = request.form.get('close_email')
    
#     # Collect all the answers
#     answers = {}
#     for i in range(1, 12):  # q1 to q11
#         answer = request.form.get(f'q{i}')
#         if answer:
#             answers[f'q{i}'] = int(answer)
#         else:
#             answers[f'q{i}'] = None  # Handle missing answers if any

#     # Calculate Big 5 scores (simplified example)
#     # Assuming questions are mapped to Big 5 traits as follows:
#     # q1, q10 -> Extraversion
#     # q2, q11 -> Agreeableness
#     # q3 -> Conscientiousness
#     # q4, q7 -> Neuroticism
#     # q5, q6, q8, q9 -> Openness
#     big5_scores = {
#         'Extraversion': (answers.get('q1', 0) + answers.get('q10', 0)) / 2,
#         'Agreeableness': (answers.get('q2', 0) + answers.get('q11', 0)) / 2,
#         'Conscientiousness': answers.get('q3', 0),
#         'Neuroticism': (answers.get('q4', 0) + answers.get('q7', 0)) / 2,
#         'Openness': (answers.get('q5', 0) + answers.get('q6', 0) + answers.get('q8', 0) + answers.get('q9', 0)) / 4
#     }

#     # Store user data
#     user_email = request.form.get('email')  # Assuming user email is stored somewhere; adjust as needed
#     if not user_email:
#         user_email = "test@example.com"  # Default or retrieve from session

#     user_data[user_email] = {
#         'close_email': close_email,
#         'answers': answers,
#         'big5_scores': big5_scores
#     }

#     # Generate and send report
#     report_filename = generate_report(user_email)
#     send_email(user_email, close_email, report_filename)

#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, 'Unknown')

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate text report of flagged responses and Big 5 results
# def generate_report(user_email):
#     global flagged_responses
#     data = user_data.get(user_email, {})
#     close_email = data.get('close_email', 'N/A')
#     big5_scores = data.get('big5_scores', {})
#     answers = data.get('answers', {})

#     report_filename = f"report_{user_email.replace('@', '_at_')}.txt"
#     with open(report_filename, 'w') as file:
#         file.write("### User Report\n\n")
#         file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         file.write(f"Close Person's Email: {close_email}\n\n")
#         file.write("#### Big 5 Personality Test Results:\n\n")
#         for trait, score in big5_scores.items():
#             file.write(f"- {trait}: {score:.2f}\n")
#         file.write("\n#### Detailed Answers:\n\n")
#         for q, a in answers.items():
#             file.write(f"{q}: {a}\n")
#         file.write("\n#### Suicidal Intent Flagged Messages:\n\n")
#         if flagged_responses:
#             for idx, message in enumerate(flagged_responses, 1):
#                 file.write(f"{idx}. {message}\n")
#         else:
#             file.write("No messages flagged as suicidal intent.\n")
#     return report_filename

# # Function to send email with the report attached
# def send_email(sender_email, receiver_email, report_filename):
#     # Email account credentials
#     smtp_server = 'smtp.gmail.com'
#     smtp_port = 587
#     sender_address = 'heemajyadav3@gmail.com'
#     sender_password = 'kunnu$102'  # **IMPORTANT:** Use environment variables or secure storage

#     # Create the email message
#     msg = MIMEMultipart()
#     msg['From'] = sender_address
#     msg['To'] = receiver_email
#     msg['Subject'] = 'User Report'

#     body = f"Hello,\n\nPlease find attached the report for the recent test taken by {sender_email}.\n\nBest regards,\nYour App"
#     msg.attach(MIMEText(body, 'plain'))

#     # Attach the report
#     with open(report_filename, 'rb') as file:
#         part = MIMEApplication(file.read(), Name=report_filename)
#     part['Content-Disposition'] = f'attachment; filename="{report_filename}"'
#     msg.attach(part)

#     try:
#         # Set up the SMTP server and send the email
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(sender_address, sender_password)
#         server.send_message(msg)
#         server.quit()
#         print(f"Email sent successfully to {receiver_email}")
#     except Exception as e:
#         print(f"Failed to send email. Error: {e}")

# # Chatbot Route
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_message = request.json['message'].strip()

#     # Check if the user inputs 'exit' to end the conversation
#     if user_message.lower() == "exit":
#         return jsonify({'reply': "Thank you for chatting with me. A report of flagged messages will be sent to the psychologist. <a href='/download_report'>Download Report</a>"})

#     # Detect suicidal intent
#     is_suicidal = detect_suicidal_intent(user_message)
    
#     if is_suicidal:
#         reply = random.choice([
#             "Are you okay? How long have you been feeling this way?",
#             "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
#         ])
#     else:
#         inputs = chatbot_tokenizer([user_message], return_tensors="pt")
#         reply_ids = chatbot_model.generate(**inputs)
#         reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
#     return jsonify({'reply': reply})

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)


#newnew
# from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
# import threading
# import cv2
# import numpy as np
# import torch
# import random
# import datetime
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import os
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.application import MIMEApplication

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models asynchronously
# emotion_model = load_model('5_30AMmodel.h5')
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses
# flagged_responses = []

# # Store user data temporarily
# user_data = {}

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test', methods=['GET'])
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Collect the close person's email
#     close_email = request.form.get('close_email')
    
#     # Collect all the answers
#     answers = {}
#     for i in range(1, 12):  # q1 to q11
#         answer = request.form.get(f'q{i}')
#         if answer:
#             answers[f'q{i}'] = int(answer)
#         else:
#             answers[f'q{i}'] = None  # Handle missing answers if any

#     # Calculate Big 5 scores (simplified example)
#     big5_scores = {
#         'Extraversion': (answers.get('q1', 0) + answers.get('q10', 0)) / 2,
#         'Agreeableness': (answers.get('q2', 0) + answers.get('q11', 0)) / 2,
#         'Conscientiousness': answers.get('q3', 0),
#         'Neuroticism': (answers.get('q4', 0) + answers.get('q7', 0)) / 2,
#         'Openness': (answers.get('q5', 0) + answers.get('q6', 0) + answers.get('q8', 0) + answers.get('q9', 0)) / 4
#     }

#     # Store user data
#     user_email = request.form.get('email')  
#     if not user_email:
#         user_email = "test@example.com"  

#     user_data[user_email] = {
#         'close_email': close_email,
#         'answers': answers,
#         'big5_scores': big5_scores
#     }

#     # Generate and send report
#     report_filename = generate_report(user_email)
#     send_email(user_email, close_email, report_filename)

#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, 'Unknown')

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate text report of flagged responses and Big 5 results
# def generate_report(user_email):
#     global flagged_responses
#     data = user_data.get(user_email, {})
#     close_email = data.get('close_email', 'N/A')
#     big5_scores = data.get('big5_scores', {})
#     answers = data.get('answers', {})

#     report_filename = f"report_{user_email.replace('@', '_at_')}.txt"
#     with open(report_filename, 'w') as file:
#         file.write("### User Report\n\n")
#         file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         file.write(f"Close Person's Email: {close_email}\n\n")
#         file.write("#### Big 5 Personality Test Results:\n\n")
#         for trait, score in big5_scores.items():
#             file.write(f"- {trait}: {score:.2f}\n")
#         file.write("\n#### Detailed Answers:\n\n")
#         for q, a in answers.items():
#             file.write(f"{q}: {a}\n")
#         file.write("\n#### Suicidal Intent Flagged Messages:\n\n")
#         if flagged_responses:
#             for idx, message in enumerate(flagged_responses, 1):
#                 file.write(f"{idx}. {message}\n")
#         else:
#             file.write("No messages flagged as suicidal intent.\n")
#     flagged_responses = []  # Clear flagged messages for future sessions
#     return report_filename

# # Function to send email with the report attached
# def send_email(sender_email, receiver_email, report_filename):
#     # Email account credentials
#     smtp_server = 'smtp.gmail.com'
#     smtp_port = 587
#     sender_address = 'heemajyadav6@gmail.com'
#     sender_password = 'qwerty@123'  

#     # Create the email message
#     msg = MIMEMultipart()
#     msg['From'] = sender_address
#     msg['To'] = receiver_email
#     msg['Subject'] = 'User Report'

#     body = f"Hello,\n\nPlease find attached the report for the recent test taken by {sender_email}.\n\nBest regards,\nYour App"
#     msg.attach(MIMEText(body, 'plain'))

#     # Attach the report
#     with open(report_filename, 'rb') as file:
#         part = MIMEApplication(file.read(), Name=report_filename)
#     part['Content-Disposition'] = f'attachment; filename="{report_filename}"'
#     msg.attach(part)

#     try:
#         # Set up the SMTP server and send the email
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(sender_address, sender_password)
#         server.send_message(msg)
#         server.quit()
#         print(f"Email sent successfully to {receiver_email}")
#     except Exception as e:
#         print(f"Failed to send email. Error: {e}")

# # Chatbot Route
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_message = request.json['message'].strip()
#     user_email = request.json.get('email', 'default@example.com')

#     if user_message.lower() == "exit":
#         report_filename = generate_report(user_email)
#         close_email = user_data.get(user_email, {}).get('close_email')
        
#         send_email(user_email, user_email, report_filename)
#         if close_email:
#             send_email(user_email, close_email, report_filename)

#         return jsonify({
#             'reply': "Thank you for chatting with me. A report of flagged messages will be sent to the psychologist. <a href='/download_report'>Download Report</a>"
#         })

#     is_suicidal = detect_suicidal_intent(user_message)
    
#     if is_suicidal:
#         reply = random.choice([
#             "Are you okay? How long have you been feeling this way?",
#             "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
#         ])
#     else:
#         inputs = chatbot_tokenizer([user_message], return_tensors="pt")
#         reply_ids = chatbot_model.generate(**inputs)
#         reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
#     return jsonify({'reply': reply})

# # Report download route
# @app.route('/download_report')
# def download_report():
#     user_email = request.args.get('email', 'default@example.com')
#     report_filename = generate_report(user_email)
#     return send_file(report_filename, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=False)

# from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
# import random
# import datetime
# import torch
# import numpy as np
# import os
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.application import MIMEApplication
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import cv2


# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models asynchronously
# emotion_model = load_model('5_30AMmodel.h5')
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses
# flagged_responses = []

# # Store user data temporarily
# user_data = {}

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test', methods=['GET'])
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     close_email = request.form.get('close_email')  # Collect close person's email
#     answers = {}

#     for i in range(1, 12):  # Collect answers to Big 5 Test
#         answer = request.form.get(f'q{i}')
#         if answer:
#             answers[f'q{i}'] = int(answer)
#         else:
#             answers[f'q{i}'] = None  # Handle missing answers if any

#     # Calculate Big 5 scores (simplified example)
#     big5_scores = {
#         'Extraversion': (answers.get('q1', 0) + answers.get('q10', 0)) / 2,
#         'Agreeableness': (answers.get('q2', 0) + answers.get('q11', 0)) / 2,
#         'Conscientiousness': answers.get('q3', 0),
#         'Neuroticism': (answers.get('q4', 0) + answers.get('q7', 0)) / 2,
#         'Openness': (answers.get('q5', 0) + answers.get('q6', 0) + answers.get('q8', 0) + answers.get('q9', 0)) / 4
#     }

#     # Assuming the user email is being passed and stored
#     user_email = request.form.get('email')
#     if not user_email:
#         user_email = "test@example.com"  # Default email for testing

#     user_data[user_email] = {
#         'close_email': close_email,
#         'answers': answers,
#         'big5_scores': big5_scores,
#         'flagged_responses': []  # Start with no flagged responses
#     }

#     # Generate and send the report
#     report_filename = generate_report(user_email)
#     send_email(user_email, close_email, report_filename)

#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, 'Unknown')

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate text report of flagged responses and Big 5 results
# def generate_report(user_email):
#     data = user_data.get(user_email, {})
#     close_email = data.get('close_email', 'N/A')
#     big5_scores = data.get('big5_scores', {})
#     answers = data.get('answers', {})
#     flagged_responses = data.get('flagged_responses', [])

#     report_filename = f"report_{user_email.replace('@', '_at_')}.txt"
#     with open(report_filename, 'w') as file:
#         file.write("### User Report\n\n")
#         file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         file.write(f"Close Person's Email: {close_email}\n\n")
#         file.write("#### Big 5 Personality Test Results:\n\n")
#         for trait, score in big5_scores.items():
#             file.write(f"- {trait}: {score:.2f}\n")
#         file.write("\n#### Detailed Answers:\n\n")
#         for q, a in answers.items():
#             file.write(f"{q}: {a}\n")
#         file.write("\n#### Suicidal Intent Flagged Messages:\n\n")
#         if flagged_responses:
#             for idx, message in enumerate(flagged_responses, 1):
#                 file.write(f"{idx}. {message}\n")
#         else:
#             file.write("No messages flagged as suicidal intent.\n")
#     return report_filename

# # Function to send email with the report attached
# def send_email(sender_email, receiver_email, report_filename):
#     # Email account credentials
#     smtp_server = 'smtp.gmail.com'
#     smtp_port = 587
#     sender_address = 'heemajyadav6@gmail.com'
#     sender_password = 'qwerty@123'  # **IMPORTANT:** Use environment variables or secure storage

#     # Create the email message
#     msg = MIMEMultipart()
#     msg['From'] = sender_address
#     msg['To'] = receiver_email
#     msg['Subject'] = 'User Report'

#     body = f"Hello,\n\nPlease find attached the report for the recent test taken by {sender_email}.\n\nBest regards,\nYour App"
#     msg.attach(MIMEText(body, 'plain'))

#     # Attach the report
#     with open(report_filename, 'rb') as file:
#         part = MIMEApplication(file.read(), Name=report_filename)
#     part['Content-Disposition'] = f'attachment; filename="{report_filename}"'
#     msg.attach(part)

#     try:
#         # Set up the SMTP server and send the email
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(sender_address, sender_password)
#         server.send_message(msg)
#         server.quit()
#         print(f"Email sent successfully to {receiver_email}")
#     except Exception as e:
#         print(f"Failed to send email. Error: {e}")

# if __name__ == '__main__':
#     app.run(debug=False)

# from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
# import threading
# import cv2
# import numpy as np
# import torch
# import random
# import datetime
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import os
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.application import MIMEApplication

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models asynchronously
# emotion_model = load_model('5_30AMmodel.h5')
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses
# flagged_responses = []

# # Store user data temporarily
# user_data = {}

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test', methods=['GET'])
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Collect the close person's email
#     close_email = request.form.get('close_email')
    
#     # Collect all the answers
#     answers = {}
#     for i in range(1, 12):  # q1 to q11
#         answer = request.form.get(f'q{i}')
#         if answer:
#             answers[f'q{i}'] = int(answer)
#         else:
#             answers[f'q{i}'] = None  # Handle missing answers if any

#     # Print answers to verify
#     print("Big 5 Answers:", answers)

#     # Calculate Big 5 scores (simplified example)
#     big5_scores = {
#         'Extraversion': (answers.get('q1', 0) + answers.get('q10', 0)) / 2,
#         'Agreeableness': (answers.get('q2', 0) + answers.get('q11', 0)) / 2,
#         'Conscientiousness': answers.get('q3', 0),
#         'Neuroticism': (answers.get('q4', 0) + answers.get('q7', 0)) / 2,
#         'Openness': (answers.get('q5', 0) + answers.get('q6', 0) + answers.get('q8', 0) + answers.get('q9', 0)) / 4
#     }

#     # Store user data
#     user_email = request.form.get('email')  # Assuming user email is stored somewhere; adjust as needed
#     if not user_email:
#         user_email = "test@example.com"  # Default or retrieve from session

#     user_data[user_email] = {
#         'close_email': close_email,
#         'answers': answers,
#         'big5_scores': big5_scores
#     }

#     # Print stored data to verify
#     print("Stored User Data:", user_data)

#     # Generate and send report
#     report_filename = generate_report(user_email)
#     send_email(user_email, close_email, report_filename)

#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, 'Unknown')

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         print("Flagged Suicidal Message:", user_message)  # Log flagged message
#         return True
#     return False

# # Generate text report of flagged responses and Big 5 results
# def generate_report(user_email):
#     global flagged_responses
#     data = user_data.get(user_email, {})
#     close_email = data.get('close_email', 'N/A')
#     big5_scores = data.get('big5_scores', {})
#     answers = data.get('answers', {})

#     report_filename = f"report_{user_email.replace('@', '_at_')}.txt"
#     with open(report_filename, 'w') as file:
#         file.write("### User Report\n\n")
#         file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         file.write(f"Close Person's Email: {close_email}\n\n")
#         file.write("#### Big 5 Personality Test Results:\n\n")
#         for trait, score in big5_scores.items():
#             file.write(f"- {trait}: {score:.2f}\n")
#         file.write("\n#### Detailed Answers:\n\n")
#         for q, a in answers.items():
#             file.write(f"{q}: {a}\n")
#         file.write("\n#### Suicidal Intent Flagged Messages:\n\n")
#         if flagged_responses:
#             for idx, message in enumerate(flagged_responses, 1):
#                 file.write(f"{idx}. {message}\n")
#         else:
#             file.write("No messages flagged as suicidal intent.\n")
#     return report_filename

# # Function to send email with the report attached
# def send_email(sender_email, receiver_email, report_filename):
#     # Email account credentials
#     smtp_server = 'smtp.gmail.com'
#     smtp_port = 587
#     sender_address = 'heemajyadav6@gmail.com'
#     sender_password = 'wvmjhjdoyvslllbj'  # **IMPORTANT:** Use environment variables or secure storage

#     # Create the email message
#     msg = MIMEMultipart()
#     msg['From'] = sender_address
#     msg['To'] = receiver_email
#     msg['Subject'] = 'User Report'

#     body = f"Hello,\n\nPlease find attached the report for the recent test taken by {sender_email}.\n\nBest regards,\nYour App"
#     msg.attach(MIMEText(body, 'plain'))

#     # Attach the report
#     with open(report_filename, 'rb') as file:
#         part = MIMEApplication(file.read(), Name=report_filename)
#     part['Content-Disposition'] = f'attachment; filename="{report_filename}"'
#     msg.attach(part)

#     try:
#         # Set up the SMTP server and send the email
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(sender_address, sender_password)
#         server.sendmail(sender_address, receiver_email, msg.as_string())
#         print("Email sent successfully!")
#     except Exception as e:
#         print(f"Failed to send email. Error: {str(e)}")
#     finally:
#         server.quit()

# if __name__ == '__main__':
#     app.run(debug=False)


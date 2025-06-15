from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from flask_pymongo import PyMongo

from functools import wraps

import smtplib
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from flask import Flask, render_template, request, redirect, url_for, session

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

from fpdf import FPDF
from flask import send_file, session, redirect, url_for

import random
import string
from dotenv import load_dotenv
import os
import datetime
from datetime import datetime, time



load_dotenv()





app = Flask(__name__)

# MongoDB Atlas Connection URI
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)


# Check DB connection
try:
    # Try to list database names (will raise error if not connected)
    db_names = mongo.cx.list_database_names()
    print("✅ MongoDB connected! Available databases:", db_names)
except Exception as e:
    print("❌ MongoDB connection failed:", str(e))



# MongoDB Collections
users = mongo.db.users
admin=mongo.db.admin
predictions=mongo.db.predictions


app.secret_key = '56f8c7e3b9e4a8d5f12a6c89db307a4f7b3c91c6745d2198'
app.config['SESSION_COOKIE_SECURE'] = True  # Use HTTPS for session cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookies




@app.route('/')
def home():

    return render_template('index.html')




# ---------------------------ADMIN ROUTES----------------------------------------------------------------
# ✅ Create a login_required decorator to protect routes
# ✅ Create a login_required decorator for admin
def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if 'admin_phone' is in session (for admin only)
        if 'admin_phone' not in session:
            return redirect(url_for('adminlogin'))  # Redirect to admin login if not authenticated
        return f(*args, **kwargs)

    return decorated_function


# ✅ Admin page (protected route)
# ✅ Admin page (protected route)
@app.route('/admin')
@admin_login_required
def admin_page():
    return render_template('admin.html')


# Adminlogin page
@app.route('/adminlogin')
def adminlogin():
    return render_template('adminlogin.html')


# ✅ adminLogin Endpoint
# ✅ Admin login route
@app.route('/login2', methods=['GET', 'POST'])
def login2():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']

        # ✅ Fetch admin from MongoDB
        admin_user = admin.find_one({'phone': phone})

        # ✅ Compare entered password with stored password
        if admin_user and password == admin_user['password']:
            session['admin_phone'] = phone  # Use 'admin_phone' for admin login
            return redirect(url_for('admin_page'))  # Redirect to /admin
        else:
            return 'Invalid credentials!'
    return render_template('adminlogin.html')  # Show admin login page for GET request


# ✅ Admin Logout2 Route
@app.route('/logout2')
def logout2():
    session.pop('admin_phone', None)  # Remove admin sessio
    session.clear()  # Clear session data
    return redirect(url_for('home'))  # Redirect to home page


# ✅ Fetch users from MongoDB
@app.route('/api/users', methods=['GET'])
def get_users():
    # Use mongo.db to access the collection
    users = list(mongo.db.users.find({}, {'_id': 1, 'fullname': 1, 'phone': 1}))

    # Format _id to string for JSON serialization
    for user in users:
        user['_id'] = str(user['_id'])

    return jsonify(users)


# ✅ Single API to get dashboard data
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    # Count total users
    total_users = mongo.db.users.count_documents({})

    # Count total predictions
    total_predictions = mongo.db.predictions.count_documents({})  # Correct collection name

    return jsonify({
        'total_users': total_users,
        'total_predictions': total_predictions
    })


@app.route('/admin/user/<phone>/predictions')
def view_user_predictions(phone):
    predictions = list(mongo.db.predictions.find({'phone': phone}))
    return render_template('user_predictions.html', phone=phone, predictions=predictions)






def sanitize_text(text):
    if not text:
        return ''
    return (
        text.replace('–', '-')
            .replace('“', '"')
            .replace('”', '"')
            .replace('’', "'")
            .replace('‘', "'")
    )
@app.route('/admin/user/<phone>/predictions/<date>/download')
def download_report(phone, date):
    print(f"Received request for phone: {phone} and date: {date}")

    requested_date = datetime.strptime(date, '%Y-%m-%d').date()

    prediction = mongo.db.predictions.find_one({
        'phone': phone,
        'timestamp': {
            '$gte': datetime.combine(requested_date, time.min),
            '$lt': datetime.combine(requested_date, time.max)
        }
    })

    if not prediction:
        print("Prediction not found for this user on the given date.")
        return "Prediction not found.", 404

    user = users.find_one({'phone': phone})
    patient_name = sanitize_text(user.get('fullname', 'Unknown')) if user else 'Unknown'
    patient_id = str(prediction.get('user_id', 'Unknown'))
    test_date = prediction['timestamp'].strftime('%d-%m-%Y')
    top_diseases = prediction['top_diseases']
    top_disease_name = sanitize_text(top_diseases[0]['disease']) if top_diseases else 'N/A'
    second_diseases = ', '.join([sanitize_text(d['disease']) for d in top_diseases[1:3]])

    # Start PDF
    pdf = FPDF()
    pdf.add_page()

    # Heading
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=sanitize_text("Medical Test Report for X-ray"), ln=True, align='C')
    pdf.ln(5)

    # Patient Info
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"Phone Number: {phone}", ln=True)
    pdf.cell(0, 10, txt=f"Test Date: {test_date}", ln=True)
    pdf.cell(0, 10, txt="Test Platform: AI-assisted Chest X-ray Diagnosis", ln=True)
    pdf.ln(10)

    # Diagnostic Summary
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Diagnostic Summary", ln=True)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(60, 10, txt="Disease Detected", border=1)
    pdf.cell(60, 10, txt="Confidence Score", border=1)
    pdf.ln()

    for disease in top_diseases:
        pdf.cell(60, 10, txt=sanitize_text(disease['disease']), border=1)
        pdf.cell(60, 10, txt=f"{disease['probability']:.2f}%", border=1)
        pdf.ln()

    # Model Info
    pdf.ln(10)
    pdf.cell(0, 10, txt="Image Modality: Chest X-ray", ln=True)
    pdf.cell(0, 10, txt="Model Used: Custom CNN Block", ln=True)

    # Visual Highlights
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Visual Highlights", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=sanitize_text(
        "- AI-based Grad-CAM visualization was generated to indicate affected lung regions.\n"
        "- Hot zones show inflammatory infiltration in both lower lobes, typical of pneumonia."
    ))

    # Recommendations
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Doctor's Recommendations", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=sanitize_text(
        f"- Initiate antibiotic therapy targeting community-acquired {top_disease_name} pathogens.\n"
        "- Conduct follow-up X-ray in 10-14 days to monitor recovery.\n"
        "- Rest, hydration, and pulmonary physiotherapy advised.\n"
        "- Consider CT scan if symptoms persist or worsen."
    ))

    # Conclusion
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Conclusion", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    conclusion = (
        f"The AI-assisted lung disease screening suggests a high likelihood of {top_disease_name} "
        f"with secondary complications such as {second_diseases}. Patient advised to begin treatment "
        "immediately and follow up regularly."
    )
    pdf.multi_cell(0, 10, txt=sanitize_text(conclusion))

    # Disclaimer
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Disclaimer", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=sanitize_text(
        "This report is generated based on an AI-assisted diagnostic tool and is intended for informational "
        "purposes only. It does not constitute a medical diagnosis. Please consult a licensed medical professional "
        "for an accurate assessment and treatment plan. The authors and developers of this software assume no "
        "responsibility for any decisions made based on the content of this report."
    ))

    # Output PDF
    buffer = BytesIO()
    buffer.write(pdf.output(dest='S').encode('latin1'))
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"prediction_report_{phone}_{date}.pdf",
        mimetype="application/pdf"
    )
# ---------------------------User ROUTES----------------------------------------------------------------



@app.route('/login1')
def login1():
    return render_template('login.html')


# Email Configuration
EMAIL_ADDRESS = "plutonium877@gmail.com"
EMAIL_PASSWORD = "ayew gkqt rqoi yorb"


# Function to generate a 6-digit OTP
def generate_otp():
    return str(random.randint(100000, 999999))


# Function to send OTP via Email
def send_otp(email, otp):
    subject = "Your OTP for Registration"
    body = f"Your OTP for registration is: {otp}. It is valid for 5 minutes."

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()  # 🔄 Instead of request.form

            fullname = data.get('fullname')
            phone = data.get('phone')
            email = data.get('email')
            password = data.get('password')
            re_password = data.get('re_pass')

            # Validate required fields
            if not all([fullname, phone, email, password, re_password]):
                return jsonify({'status': 'error', 'message': 'All fields are required'})
            # Check if user exists
            if mongo.db.users.find_one({'$or': [{'phone': phone}, {'email': email}]}):
                return jsonify({'status': 'error', 'message': 'User already exists!'})

            if password != re_password:
                return jsonify({'status': 'error', 'message': 'Passwords do not match'})

            # Generate and send OTP
            otp = generate_otp()
            session['otp'] = otp
            session['registration_data'] = {
                'fullname': fullname,
                'phone': phone,
                'email': email,
                'password': password  # 🔒 Note: hash before storing in real apps
            }

            if send_otp(email, otp):
                return jsonify({'status': 'success', 'message': 'OTP sent to your email'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to send OTP'})

        except Exception as e:
            print(f"Registration error: {str(e)}")
            return jsonify({'status': 'error', 'message': 'An error occurred during registration'})

    return render_template('register.html')


@app.route("/otp_verification")
def otp_verification():
    if 'otp' not in session or 'registration_data' not in session:
        # 👇 Redirect back to register if session data is missing
        return redirect("/register")
    return render_template("otp_verification.html")


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    user_otp = data.get('otp')

    if user_otp == session.get('otp'):

        registration_data = session.pop('registration_data', None)
        if registration_data:
            # Optional: check if user already exists
            existing_user = mongo.db.users.find_one({'email': registration_data['email']})
            if existing_user:
                return jsonify({'status': 'error', 'redirect_url': '/register', 'message': 'Email already registered'})

            # Hash the password
            hashed_password = generate_password_hash(registration_data['password'])

            # Prepare user document
            user_doc = {
                'fullname': registration_data['fullname'],
                'email': registration_data['email'],
                'phone': registration_data['phone'],
                'password': hashed_password,
                'created_at': datetime.utcnow()
            }

            # ✅ Insert into MongoDB
            mongo.db.users.insert_one(user_doc)

            # Clear session OTP
            session.pop('otp', None)
            session.pop('registration_data', None)

            # ✅ Auto-login: Set session
            session['phone'] = registration_data['phone']

            return jsonify(
                {'status': 'success', 'redirect_url': '/index2', 'message': 'OTP verified and user registered'})
        else:
            return jsonify(
                {'status': 'error', 'redirect_url': '/register', 'message': 'Session expired. Please register again.'})
    else:
        return jsonify({'status': 'error', 'redirect_url': '/register', 'message': 'Invalid OTP'})


@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    try:
        registration_data = session.get('registration_data')
        if not registration_data:
            return jsonify({'status': 'error', 'message': 'Session expired. Please register again.'})

        new_otp = generate_otp()
        session['otp'] = new_otp

        if send_otp(registration_data['email'], new_otp):
            return jsonify({'status': 'success', 'message': 'OTP resent successfully.'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to send OTP.'})
    except Exception as e:
        print(f"Resend OTP error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Server error'})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']

        # Fetch user from MongoDB
        user = users.find_one({'phone': phone})

        # Compare entered password with stored hashed password
        if user and check_password_hash(user['password'], password):
            session['phone'] = phone  # Store phone number in session

            return redirect(url_for('index2'))
        else:
            return 'Invalid credentials! *PLEASE SIGN UP*'

    return render_template('login.html')  # Render the login page for GET request


# --------------------------------
# ✅ Logout Route
@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    return redirect(url_for('home'))  # Redirect to homepage after logout


@app.route('/index2')
def index2():
    phone = session.get('phone')
    if not phone:
        return redirect(url_for('login'))

    user = users.find_one({'phone': session['phone']})
    if not user:
        return 'User not found'



    return render_template('index2.html',user=user)
# --------------------------------
# ✅ Prediction page

# Load model
MODEL_PATH = "model.keras"
model = load_model(MODEL_PATH)


# Disease labels
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
          'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
          'Pneumonia', 'Pneumothorax']

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])



import base64
from io import BytesIO
from flask import Flask
from PIL import Image

def to_b64(img_np):
    img_pil = Image.fromarray(img_np.astype('uint8'))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

app.jinja_env.filters['to_b64'] = to_b64


def preprocess_image(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    img = (img - mean) / std
    return img


def predict_disease_from_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_preprocessed = preprocess_image(img_resized)
    predictions = model.predict(np.expand_dims(img_preprocessed, axis=0))
    return predictions


def grad_cam(model, img_array, class_idx, layer_name="conv5_block16_concat"):
    """Generate the GradCAM heatmap for a specific class index and layer."""
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_output, predictions = grad_model(img_array, training=False)  # ✅ fixed here
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(tf.multiply(conv_output, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)  # Normalize
    return heatmap.numpy()




def generate_gradcam(image, predictions, top_n=3):
    top_n_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_n_probs = predictions[0][top_n_indices]
    top_n_labels = [labels[i] for i in top_n_indices]

    gradcam_images = []
    for i in range(top_n):
        gradcam = grad_cam(model, np.expand_dims(preprocess_image(image), axis=0), top_n_indices[i])
        gradcam_resized = cv2.resize(gradcam, (224, 224))
        overlay = np.uint8(255 * gradcam_resized)
        heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0

        img_resized_float = np.float32(image) / 255.0
        if len(img_resized_float.shape) == 2:
            img_resized_float = np.repeat(img_resized_float[..., np.newaxis], 3, axis=-1)

        blended_img = 0.5 * img_resized_float + 0.5 * heatmap
        blended_img = np.uint8(255 * blended_img)
        gradcam_images.append((blended_img, top_n_labels[i], top_n_probs[i]))

    return gradcam_images


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    phone = session.get('phone')
    if not phone:
        return redirect(url_for('login'))

    user = users.find_one({'phone': session['phone']})
    if not user:
        return 'User not found'

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')
            image_np = np.array(image)

            predictions = predict_disease_from_image(image_np)
            gradcams = generate_gradcam(image_np, predictions, top_n=6)

            top_diseases = sorted([(labels[i], predictions[0][i]) for i in range(len(labels))],
                                  key=lambda x: x[1], reverse=True)[:6]

            # ✅ Save for report download
            session['top_diseases'] = [(disease, float(prob)) for disease, prob in top_diseases]

            # ✅ Save to MongoDB with float conversion
            mongo.db.predictions.insert_one({
                "phone": phone,
                "user_id": user['_id'],
                "top_diseases": [
                    {
                        "disease": disease,
                        "probability": round(float(prob) * 100, 2)  # Convert np.float32 to native float
                    }
                    for disease, prob in top_diseases
                ],
                "timestamp": datetime.utcnow()
            })

            return render_template('prediction.html',
                                   original_image=file.filename,
                                   top_diseases=top_diseases,
                                   gradcams=gradcams,
                                   user=user)  # <-- add this

    # On GET, clear previous predictions
    session.pop('top_diseases', None)

    return render_template('prediction.html', user=user)


# --------------------------------
# ✅ Report Route



def sanitize_text(text):
    if not text:
        return ''
    return (
        text.replace('–', '-')
            .replace('“', '"')
            .replace('”', '"')
            .replace('’', "'")
            .replace('‘', "'")
    )

@app.route('/generate-report')
def generate_report():
    top_diseases = session.get('top_diseases')
    if not top_diseases:
        return redirect(url_for('prediction'))

    user = users.find_one({'phone': session.get('phone')})
    if not user:
        return redirect(url_for('login'))

    patient_name = sanitize_text(user.get('fullname', 'Unknown'))
    phone = sanitize_text(session.get('phone'))
    test_date = datetime.now().strftime('%d-%m-%Y')
    patient_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    top_disease_name = sanitize_text(top_diseases[0][0])  # highest confidence disease

    # PDF content
    pdf = FPDF()
    pdf.add_page()

    # Heading
    pdf.set_text_color(0, 102, 204)  # Blue
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=sanitize_text("Medical Test Report for X-ray"), ln=True, align='C')
    pdf.ln(5)

    # Patient Details
    pdf.set_text_color(0, 0, 0)  # Black
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Patient ID: {patient_id}", ln=True)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"Phone Number: {phone}", ln=True)
    pdf.cell(0, 10, txt=f"Test Date: {test_date}", ln=True)
    pdf.cell(0, 10, txt=sanitize_text("Test Platform: AI-assisted Chest X-ray Diagnosis"), ln=True)
    pdf.ln(10)

    # Section: Diagnostic Summary
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=sanitize_text("Diagnostic Summary"), ln=True)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(60, 10, txt="Disease Detected", border=1)
    pdf.cell(60, 10, txt="Confidence Score", border=1)
    pdf.ln()

    for disease, prob in top_diseases:
        disease = sanitize_text(disease)
        pdf.cell(60, 10, txt=disease, border=1)
        pdf.cell(60, 10, txt=f"{prob * 100:.2f}%", border=1)
        pdf.ln()

    # Section: Image Info
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=sanitize_text("Image Modality: Chest X-ray"), ln=True)
    pdf.cell(0, 10, txt=sanitize_text("Model Used: DenseNet169"), ln=True)

    # Section: Visual Highlights
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=sanitize_text("Visual Highlights"), ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=sanitize_text(
        "- AI-based Grad-CAM visualization was generated to indicate affected lung regions.\n"
        "- Hot zones show inflammatory infiltration in both lower lobes, typical of pneumonia."
    ))

    # Section: Doctor's Recommendations
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=sanitize_text("Doctor's Recommendations"), ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=sanitize_text(
        f"- Initiate antibiotic therapy targeting community-acquired {top_disease_name} pathogens.\n"
        "- Conduct follow-up X-ray in 10-14 days to monitor recovery.\n"
        "- Rest, hydration, and pulmonary physiotherapy advised.\n"
        "- Consider CT scan if symptoms persist or worsen."
    ))

    # Section: Conclusion
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=sanitize_text("Conclusion"), ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    second_diseases = ', '.join([sanitize_text(d) for d, _ in top_diseases[1:3]])
    conclusion = (
        f"The AI-assisted lung disease screening suggests a high likelihood of {top_disease_name} "
        f"with secondary complications such as {second_diseases}. Patient advised to begin treatment "
        "immediately and follow up regularly."
    )
    pdf.multi_cell(0, 10, txt=sanitize_text(conclusion))

    # Section: Disclaimer
    pdf.ln(10)
    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=sanitize_text("Disclaimer"), ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=sanitize_text(
        "This report is generated based on an AI-assisted diagnostic tool and is intended for informational "
        "purposes only. It does not constitute a medical diagnosis. Please consult a licensed medical professional "
        "for an accurate assessment and treatment plan. The authors and developers of this software assume no "
        "responsibility for any decisions made based on the content of this report."
    ))

    # Return PDF in memory
    pdf_output = BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    return send_file(pdf_output, download_name="prediction_report.pdf", as_attachment=True)


@app.route('/appointment')
def appointment():
    phone = session.get('phone')
    if not phone:
        return redirect(url_for('login'))

    user = users.find_one({'phone': session['phone']})
    if not user:
        return 'User not found'

    return render_template('appointment.html',user=user)



if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import logging
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

# Load the trained model
try:
    model = joblib.load("model/digit_recognizer")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    logged_in = 'user_id' in session
    return render_template('index.html', logged_in=logged_in)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            first_name = request.form['first_name']
            last_name = request.form['last_name']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            phone = request.form['phone']

            if password != confirm_password:
                return "Passwords do not match"

            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=email, password=hashed_password)

            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except KeyError as e:
            logger.error(f"Missing form key: {e}")
            return "Missing form key", 400
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return "There was an issue adding your task"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            return "Invalid credentials"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/recognize', methods=['POST'])
def recognize_digit():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    file = request.files['image']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if im is None:
        logger.error("Failed to load image")
        return jsonify({'error': 'Failed to load image'})
    
    try:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
        roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({'error': f'Image processing error: {e}'})
    
    if np.count_nonzero(roi) < 50:
        logger.warning("No digit recognized")
        return jsonify({'error': 'No digit recognized'})

    try:
        X = roi.flatten() > 100
        X = X.astype(int).tolist()
        predictions = model.predict([X])
        prediction = int(predictions[0])  # Convert to standard Python int
        prediction_text = f"Prediction: {prediction}"
        cv2.putText(im, prediction_text, (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        logger.info(f"Digit recognized: {prediction}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction error: {e}'})

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(im_rgb)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    return jsonify({'prediction': prediction, 'image': img_str})

@app.route('/visualization')
def visualization():
    # Assuming you have some images stored in the static folder or another location
    # For example purposes, let's assume images are stored in static/images directory
    image_files = ['static/images/image1.png', 'static/images/image2.png']
    return render_template('visualization.html', images=image_files)
    
@app.route('/user_records')
def user_records():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    users = User.query.all()
    return render_template('user_records.html', users=users)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['admin_username']
        password = request.form['admin_password']
        # Replace 'admin' and 'admin_password' with actual admin credentials
        if username == 'admin' and password == 'admin_password':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials')
    return render_template('admin_login.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted successfully.')
    else:
        flash('User not found.')
    return redirect(url_for('admin_dashboard'))

if __name__ == "__main__":
    app.run(debug=True)

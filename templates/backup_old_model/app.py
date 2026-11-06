from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader

import joblib
import numpy as np
import os

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

app.config["SECRET_KEY"] = "fake_news_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Load the trained model
print("Loading model...")
model = joblib.load("model.pkl")
print("Model loaded.")

print("Loading vectorizer...")
vectorizer = joblib.load("vectorizer.pkl")
print("Vectorizer loaded.")

# ======================
# Database Table
# ======================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ======================
# Detection History Table
# ======================
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    news = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

# ======================
# Home
# ======================
@app.route("/")
def home():
    return redirect(url_for("login"))


# ======================
# Register
# ======================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        existing = User.query.filter_by(username=username).first()

        if existing:
            flash("Username already exists.")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)

        user = User(username=username, password=hashed)

        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.")

        return redirect(url_for("login"))

    return render_template("register.html")


# ======================
# Login
# ======================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):

            session["user"] = user.id
            session["username"] = user.username

            return redirect(url_for("dashboard"))

        flash("Invalid username or password.")

    return render_template("login.html")

# ======================
# Dashboard
# ======================
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    total = History.query.filter_by(user_id=session["user"]).count()

    real = History.query.filter_by(
        user_id=session["user"],
        prediction="✅ Real News"
    ).count()

    fake = History.query.filter_by(
        user_id=session["user"],
        prediction="🚨 Fake News"
    ).count()

    records = History.query.filter_by(
        user_id=session["user"]
    ).all()

    if records:
        avg_confidence = round(
            sum(r.confidence for r in records) / len(records),
            2
        )
    else:
        avg_confidence = 0

    recent = History.query.filter_by(
        user_id=session["user"]
    ).order_by(
        History.id.desc()
    ).limit(5).all()

    return render_template(
        "dashboard.html",
        total=total,
        real=real,
        fake=fake,
        avg_confidence=avg_confidence,
        recent=recent
    )
@app.route("/detect", methods=["GET", "POST"])
def detect():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    news = ""

    if request.method == "POST":

        news = request.form["news"]

        vector = vectorizer.transform([news])

        prediction_value = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        confidence = round(np.max(probability) * 100, 2)

        if prediction_value == 0:
            prediction = "🚨 Fake News"
        else:
            prediction = "✅ Real News"

        history = History(
            user_id=session["user"],
            news=news,
            prediction=prediction,
            confidence=float(confidence)
        )

        db.session.add(history)
        db.session.commit()

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=confidence,
        news=news
    )

    # ======================
# Upload File
# ======================
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    extracted_text = ""

    if request.method == "POST":

        if "file" not in request.files:
            flash("Please choose a file.")
            return redirect(url_for("upload"))

        file = request.files["file"]

        if file.filename == "":
            flash("Please choose a file.")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        extension = os.path.splitext(filename)[1].lower()

        try:

            if extension == ".txt":

                extracted_text = file.read().decode("utf-8")

            elif extension == ".pdf":

                reader = PdfReader(file)

                for page in reader.pages:
                    text = page.extract_text()

                    if text:
                        extracted_text += text + "\n"

            else:

                flash("Only TXT and PDF files are supported.")
                return redirect(url_for("upload"))

            if extracted_text.strip():

                vector = vectorizer.transform([extracted_text])

                prediction_value = model.predict(vector)[0]

                probability = model.predict_proba(vector)

                confidence = round(np.max(probability) * 100, 2)

                if prediction_value == 0:
                    prediction = "🚨 Fake News"
                else:
                    prediction = "✅ Real News"

                history = History(
                    user_id=session["user"],
                    news=extracted_text[:500],
                    prediction=prediction,
                    confidence=float(confidence)
                )

                db.session.add(history)
                db.session.commit()

            else:

                flash("No readable text found in the file.")

        except Exception as e:

            flash(f"Error reading file: {e}")

    return render_template(
        "upload.html",
        prediction=prediction,
        confidence=confidence
    )
# ======================
# Upload Image
# ======================
@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    extracted_text = ""
    image_path = None

    if request.method == "POST":

        if "image" not in request.files:
            flash("Please select an image.")
            return redirect(url_for("upload_image"))

        image = request.files["image"]

        if image.filename == "":
            flash("Please select an image.")
            return redirect(url_for("upload_image"))

        try:

            # Save uploaded image
            filename = secure_filename(image.filename)

            upload_folder = os.path.join("static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)

            save_path = os.path.join(upload_folder, filename)
            image.save(save_path)

            image_path = "/" + save_path.replace("\\", "/")

            # OCR
            img = Image.open(save_path)
            extracted_text = pytesseract.image_to_string(img)

            if extracted_text.strip() != "":

                vector = vectorizer.transform([extracted_text])

                prediction_value = model.predict(vector)[0]

                probability = model.predict_proba(vector)

                confidence = round(np.max(probability) * 100, 2)

                if prediction_value == 0:
                    prediction = "🚨 Fake News"
                else:
                    prediction = "✅ Real News"

                history = History(
                    user_id=session["user"],
                    news=extracted_text[:500],
                    prediction=prediction,
                    confidence=float(confidence)
                )

                db.session.add(history)
                db.session.commit()

            else:
                flash("No text was found in the image.")

        except Exception as e:
            flash(f"Error: {e}")

    return render_template(
        "upload_image.html",
        prediction=prediction,
        confidence=confidence,
        extracted_text=extracted_text,
        image_path=image_path
    )   
# ======================
# History
# ======================
@app.route("/history")
def history():

    if "user" not in session:
        return redirect(url_for("login"))

    records = History.query.filter_by(
        user_id=session["user"]
    ).order_by(History.id.desc()).all()

    return render_template(
        "history.html",
        records=records
    )


# ======================
# Delete History
# ======================
@app.route("/delete_history/<int:id>")
def delete_history(id):

    if "user" not in session:
        return redirect(url_for("login"))

    record = History.query.get_or_404(id)

    # Make sure users can only delete their own records
    if record.user_id != session["user"]:
        flash("Unauthorized action.")
        return redirect(url_for("history"))

    db.session.delete(record)
    db.session.commit()

    flash("History deleted successfully.")

    return redirect(url_for("history"))

    

# ======================
# About
# ======================
@app.route("/about")
def about():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("about.html")

# ======================
# Settings
# ======================
@app.route("/settings", methods=["GET", "POST"])
def settings():

    if "user" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user"])

    if request.method == "POST":

        new_username = request.form["username"]
        new_password = request.form["password"]

        if new_username.strip():
            user.username = new_username
            session["username"] = new_username

        if new_password.strip():
            user.password = generate_password_hash(new_password)

        db.session.commit()

        flash("Account updated successfully.")

        return redirect(url_for("settings"))

    return render_template("settings.html", user=user)

# ======================
# Logout
# ======================
@app.route("/logout")
def logout():

    session.pop("user", None)
    session.pop("username", None)

    return redirect(url_for("login"))

    # ======================
# Start Application
# ======================
if __name__ == "__main__":

    print("Creating database...")

    with app.app_context():
        db.create_all()

    print("Starting Flask server...")

    app.run(debug=True)

    print("This line should never print.")

   

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

import webbrowser
import threading
import joblib
import numpy as np
import os
import requests

from dotenv import load_dotenv
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Load environment variables from .env
load_dotenv()

# Tesseract OCR installation path on Windows.
# Change this path only if Tesseract is installed elsewhere.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =====================================================
# FLASK CONFIGURATION
# =====================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv(
    "FLASK_SECRET_KEY",
    "fake_news_secret_key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# =====================================================
# LOAD MACHINE-LEARNING MODEL
# =====================================================

print("Loading model...")
model = joblib.load("model.pkl")
print("Model loaded.")

print("Loading vectorizer...")
vectorizer = joblib.load("vectorizer.pkl")
print("Vectorizer loaded.")

# =====================================================
# DATABASE TABLES
# =====================================================

class User(db.Model):
    # Kept for compatibility with the existing database.
    # Login and registration are not required.
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    news = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)


def get_default_user_id():
    """Get a local database user for saving history without login."""
    user = User.query.order_by(User.id.asc()).first()

    if user is None:
        user = User(
            username="local_dashboard",
            password=generate_password_hash(os.urandom(24).hex())
        )

        db.session.add(user)
        db.session.commit()

    return user.id


# Create database tables if they do not exist.
with app.app_context():
    db.create_all()
    get_default_user_id()

# =====================================================
# HOME - OPEN DASHBOARD IMMEDIATELY
# =====================================================

@app.route("/")
def home():
    return redirect(url_for("detect"))


# =====================================================
# DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():

    total = History.query.count()

    real = History.query.filter_by(
        prediction="✅ Real News"
    ).count()

    fake = History.query.filter_by(
        prediction="🚨 Fake News"
    ).count()

    records = History.query.all()

    if records:
        avg_confidence = round(
            sum(record.confidence for record in records)
            / len(records),
            2
        )
    else:
        avg_confidence = 0

    recent = History.query.order_by(
        History.id.desc()
    ).all()

    return render_template(
        "dashboard.html",
        total=total,
        real=real,
        fake=fake,
        avg_confidence=avg_confidence,
        recent=recent
    )


# =====================================================
# ONLINE NEWS SEARCH
# =====================================================

def search_online_news(title, news):
    """Search NewsAPI for articles related to submitted news."""

    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        return [], "Online search unavailable: API key not found."

    query = (title or "").strip()

    if not query:
        query = " ".join((news or "").split()[:12])

    if not query:
        return [], "No headline or article text was provided."

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": api_key
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        data = response.json()

    except (requests.RequestException, ValueError):
        return [], "Could not connect to the online news search service."

    if response.status_code != 200 or data.get("status") != "ok":
        message = data.get(
            "message",
            "The online search request failed."
        )

        return [], f"Online search unavailable: {message}"

    articles = []

    for item in data.get("articles", []):
        articles.append({
            "title": item.get("title") or "Untitled article",
            "source": (
                (item.get("source") or {}).get("name", "Unknown source")
            ),
            "url": item.get("url") or "",
            "published_at": item.get("publishedAt") or "",
            "description": item.get("description") or ""
        })

    if articles:
        message = (
            "Related online articles were found. "
            "Review the sources to assess the claim."
        )
    else:
        message = (
            "No matching articles were found. "
            "This does not automatically mean the news is fake."
        )

    return articles, message


# =====================================================
# DETECT NEWS
# =====================================================

@app.route("/detect", methods=["GET", "POST"])
def detect():

    prediction = None
    confidence = None
    news = ""
    title = ""
    online_articles = []
    online_message = None

    if request.method == "POST":

        title = request.form.get("title", "").strip()
        news = request.form.get("news", "").strip()

        if not news:
            prediction = "Please enter a news article."

        else:
            combined_text = f"{title} {news}".strip()

            vector = vectorizer.transform([combined_text])

            prediction_value = int(model.predict(vector)[0])

            probabilities = model.predict_proba(vector)[0]

            confidence = round(
                float(np.max(probabilities)) * 100,
                2
            )

            if prediction_value == 0:
                prediction = "🚨 Fake News"
            else:
                prediction = "✅ Real News"

            # Search for related online articles.
            online_articles, online_message = search_online_news(
                title,
                news
            )

            # Save the prediction in the database.
            history_record = History(
                user_id=get_default_user_id(),
                news=news,
                prediction=prediction,
                confidence=float(confidence)
            )

            db.session.add(history_record)
            db.session.commit()

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=confidence,
        news=news,
        title=title,
        online_articles=online_articles,
        online_message=online_message
    )


# =====================================================
# UPLOAD PDF OR TXT FILE
# =====================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    prediction = None
    confidence = None
    extracted_text = ""
    online_articles = []
    online_message = None

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

                extracted_text = file.read().decode(
                    "utf-8",
                    errors="replace"
                )

            elif extension == ".pdf":

                reader = PdfReader(file)

                extracted_text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

            else:
                flash("Only TXT and PDF files are supported.")
                return redirect(url_for("upload"))

            if extracted_text.strip():

                vector = vectorizer.transform([extracted_text])

                prediction_value = int(model.predict(vector)[0])

                probabilities = model.predict_proba(vector)[0]

                confidence = round(
                    float(np.max(probabilities)) * 100,
                    2
                )

                if prediction_value == 0:
                    prediction = "🚨 Fake News"
                else:
                    prediction = "✅ Real News"

                online_articles, online_message = search_online_news(
                    "",
                    extracted_text
                )

                history_record = History(
                    user_id=get_default_user_id(),
                    news=extracted_text[:500],
                    prediction=prediction,
                    confidence=float(confidence)
                )

                db.session.add(history_record)
                db.session.commit()

            else:
                flash("No readable text found in the file.")

        except Exception:
            db.session.rollback()

            app.logger.exception(
                "File upload or processing failed"
            )

            flash(
                "Error processing the file. "
                "Check that it is a valid TXT or PDF file."
            )

    return render_template(
        "upload.html",
        prediction=prediction,
        confidence=confidence,
        online_articles=online_articles,
        online_message=online_message
    )


# =====================================================
# UPLOAD IMAGE AND EXTRACT TEXT USING OCR
# =====================================================

@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():

    prediction = None
    confidence = None
    extracted_text = ""
    image_path = None
    online_articles = []
    online_message = None

    if request.method == "POST":

        if "image" not in request.files:
            flash("Please select an image.")
            return redirect(url_for("upload_image"))

        image = request.files["image"]

        if image.filename == "":
            flash("Please select an image.")
            return redirect(url_for("upload_image"))

        try:

            filename = secure_filename(image.filename)

            extension = os.path.splitext(filename)[1].lower()

            allowed_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff"
            }

            if extension not in allowed_extensions:
                flash(
                    "Please upload a JPG, JPEG, PNG, BMP, or TIFF image."
                )
                return redirect(url_for("upload_image"))

            upload_folder = os.path.join("static", "uploads")

            os.makedirs(upload_folder, exist_ok=True)

            save_path = os.path.join(upload_folder, filename)

            image.save(save_path)

            image_path = "/" + save_path.replace("\\", "/")

            # Improve OCR accuracy by enlarging and processing the image.
            with Image.open(save_path) as img:

                img = img.convert("RGB")

                width, height = img.size

                img = img.resize(
                    (
                        max(1, width * 2),
                        max(1, height * 2)
                    ),
                    Image.Resampling.LANCZOS
                )

                gray = ImageOps.grayscale(img)

                gray = ImageEnhance.Contrast(gray).enhance(1.8)

                gray = gray.filter(ImageFilter.SHARPEN)

                threshold_img = gray.point(
                    lambda pixel: 255 if pixel > 170 else 0
                )

                ocr_results = []

                for ocr_img in (gray, threshold_img):

                    for page_mode in (6, 11):

                        text_result = pytesseract.image_to_string(
                            ocr_img,
                            config=f"--oem 3 --psm {page_mode}"
                        )

                        if text_result.strip():
                            ocr_results.append(text_result.strip())

                extracted_text = max(
                    ocr_results,
                    key=len,
                    default=""
                )

            if extracted_text.strip():

                # The model classifies extracted text, not image pixels.
                vector = vectorizer.transform([extracted_text])

                prediction_value = int(model.predict(vector)[0])

                probabilities = model.predict_proba(vector)[0]

                confidence = round(
                    float(np.max(probabilities)) * 100,
                    2
                )

                if prediction_value == 0:
                    prediction = "🚨 Fake News"
                else:
                    prediction = "✅ Real News"

                online_articles, online_message = search_online_news(
                    "",
                    extracted_text
                )

                history_record = History(
                    user_id=get_default_user_id(),
                    news=extracted_text[:500],
                    prediction=prediction,
                    confidence=float(confidence)
                )

                db.session.add(history_record)
                db.session.commit()

            else:
                flash("No text was found in the image.")

        except Exception:
            db.session.rollback()

            app.logger.exception("Image processing failed")

            flash(
                "Error processing the image. "
                "Check the image and Tesseract OCR installation."
            )

    return render_template(
        "upload_image.html",
        prediction=prediction,
        confidence=confidence,
        extracted_text=extracted_text,
        image_path=image_path,
        online_articles=online_articles,
        online_message=online_message
    )


# =====================================================
# DETECTION HISTORY
# =====================================================

@app.route("/history")
def history():

    records = History.query.order_by(
        History.id.desc()
    ).all()

    return render_template(
        "history.html",
        records=records
    )


# =====================================================
# DELETE HISTORY RECORD
# =====================================================

@app.route("/delete_history/<int:id>", methods=["GET", "POST"])
def delete_history(id):

    record = History.query.get_or_404(id)

    db.session.delete(record)
    db.session.commit()

    flash("History deleted successfully.")

    return redirect(url_for("history"))


# =====================================================
# ABOUT PAGE
# =====================================================

@app.route("/about")
def about():
    return render_template("about.html")


# =====================================================
# LEGACY ACCOUNT ROUTES
# =====================================================
# These endpoints remain available because existing templates
# may refer to their names with url_for().
# They do not display login, registration, or settings pages.

@app.route("/settings", methods=["GET", "POST"])
def settings():
    return redirect(url_for("detect"))


@app.route("/login", methods=["GET", "POST"])
def login():
    return redirect(url_for("detect"))


@app.route("/register", methods=["GET", "POST"])
def register():
    return redirect(url_for("detect"))


@app.route("/logout")
def logout():
    return redirect(url_for("detect"))


# =====================================================
# START APPLICATION
# =====================================================

if __name__ == "__main__":

    threading.Timer(
        1.5,
        lambda: webbrowser.open("http://127.0.0.1:5000/")
    ).start()

    app.run(
        debug=True,
        use_reloader=False
    )

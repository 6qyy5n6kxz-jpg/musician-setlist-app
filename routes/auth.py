# filepath: /Users/devinfrank/setlist-genie/routes/auth.py
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.security import check_password_hash

from models import db, User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/register/", methods=["GET", "POST"])
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("list_songs"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        if not username or not email or not password:
            flash("All fields are required.", "danger")
        elif password != password2:
            flash("Passwords do not match.", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("auth.login"))
    return render_template('auth/register.html')

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("list_songs"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully.", "success")
            next_url = request.args.get("next") or ""
            # Only follow same-site relative paths to prevent open redirects
            if not next_url.startswith("/") or next_url.startswith("//") or "\\" in next_url:
                next_url = url_for("list_songs")
            return redirect(next_url)
        else:
            flash("Invalid username or password.", "danger")
    return render_template('auth/login.html')

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))

# --- Change password (must be logged in and know the current password) ---
# Forgotten passwords are reset by an admin: `flask --app app set-password <username>`
@auth_bp.route("/reset-password", methods=["GET", "POST"])
@login_required
def reset_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        if not current or not password:
            flash("All fields are required.", "danger")
        elif not current_user.check_password(current):
            flash("Current password is incorrect.", "danger")
        elif password != password2:
            flash("Passwords do not match.", "danger")
        elif len(password) < 8:
            flash("New password must be at least 8 characters.", "danger")
        else:
            current_user.set_password(password)
            db.session.commit()
            flash("Password changed.", "success")
            return redirect(url_for("list_songs"))
    return render_template('auth/reset.html')

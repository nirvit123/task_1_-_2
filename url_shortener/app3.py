from flask import Flask, render_template, request, redirect
from models import db, URL
import validators
import string
import random

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///url_shortener.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def generate_short_code(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

with app.app_context():
    db.create_all()


@app.route('/', methods=['GET', 'POST'])
def home():
    short_url = None
    error = None

    if request.method == 'POST':
        original_url = request.form['url']

        if not validators.url(original_url):
            error = "Invalid URL. Please enter a valid URL."
        else:
            short_code = generate_short_code()
            new_url = URL(original_url=original_url, short_code=short_code)
            db.session.add(new_url)
            db.session.commit()
            short_url = request.host_url + short_code

    return render_template('index.html', short_url=short_url, error=error)

@app.route('/<short_code>')
def redirect_url(short_code):
    url = URL.query.filter_by(short_code=short_code).first_or_404()
    return redirect(url.original_url)

@app.route('/history')
def history():
    urls = URL.query.all()
    return render_template('history.html', urls=urls)

if __name__ == '__main__':
    app.run(debug=True)

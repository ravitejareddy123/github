from flask import Flask, request, render_template_string
import logging
from datetime import datetime

app = Flask(__name__)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Simple login page HTML
LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .login-container { width: 300px; padding: 20px; border: 1px solid #ccc; border-radius: 5px; }
        .login-container h2 { text-align: center; }
        .login-container input { width: 100%; padding: 8px; margin: 10px 0; }
        .login-container button { width: 100%; padding: 10px; background-color: #2563eb; color: white; border: none; border-radius: 5px; }
        .login-container button:hover { background-color: #1e40af; }
        .error { color: red; text-align: center; }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>Login</h2>
        <form method="POST" action="/login">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def hello():
    start_time = datetime.now()
    response = 'Hello from Microservice!'
    logger.info(f"Request processed successfully, response time: {(datetime.now() - start_time).total_seconds() * 1000:.2f}ms")
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    start_time = datetime.now()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password':
            logger.info(f"Login successful for user: {username}, response time: {(datetime.now() - start_time).total_seconds() * 1000:.2f}ms")
            return 'Login successful!'
        else:
            logger.error(f"Login failed for user: {username}")
            return render_template_string(LOGIN_PAGE, error='Invalid credentials')
    logger.info(f"Login page accessed, response time: {(datetime.now() - start_time).total_seconds() * 1000:.2f}ms")
    return render_template_string(LOGIN_PAGE)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response, send_from_directory
import sqlite3
from groq import Groq
from typing import List
import json
from datetime import datetime
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from functools import wraps
from src.text_extractor import extract_text_from_pdf
from src.question_maker import QuestionMaker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict
from src.chatbot import Chatbot

app = Flask(__name__)
app.secret_key = "supersecretkey"
DATABASE = 'users.db'

# Folder to store uploaded PDFs
UPLOAD_FOLDER = 'knowledgebase'
import shutil
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
if os.path.exists("chat_logs"):
    shutil.rmtree("chat_logs")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

# Check if the file is a valid PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Add this decorator definition before your routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in first.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials. Please try again.", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()

            flash("Account created successfully! You can now log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please try a different one.", "danger")
            return redirect(url_for("signup"))

    return render_template("signup.html")
@app.route("/question", methods=["GET", "POST"])
def question():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash(f"File '{filename}' uploaded successfully!", "success")
        else:
            flash("Only PDF files are allowed.", "danger")
    return render_template("questions.html")
import os 
import dotenv 
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)
CHAT_LOGS_FOLDER = 'chat_logs'
if not os.path.exists(CHAT_LOGS_FOLDER):
    os.makedirs(CHAT_LOGS_FOLDER)

CONVERSATION_FILE = f"{CHAT_LOGS_FOLDER}/conversation_history.json"
def load_conversation_history():
    try:
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading conversation history: {e}")
    return []

def save_to_conversation_history(message: dict):
    try:
        history = load_conversation_history()
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": message["role"],
            "content": message["content"]
        })
        
        with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error saving to conversation history: {e}")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        if "message" in request.form:
            try:
                user_message = request.form["message"]
                save_to_conversation_history({
                    "role": "user",
                    "content": user_message
                })
                
                def generate_response():
                    try:
                        # Load full conversation history for context
                        history = load_conversation_history()
                        
                        # Prepare messages for Groq, including conversation history
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."}
                        ]
                        
                        # Add recent conversation history (last 10 messages)
                        for msg in history[-10:]:
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        
                        # Get streaming response from Groq
                        completion = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=messages,
                            temperature=1,
                            max_tokens=1024,
                            top_p=1,
                            stream=True,
                            stop=None
                        )
                        
                        # Initialize full response to save later
                        full_response = ""
                        
                        # Stream the response
                        for chunk in completion:
                            content = chunk.choices[0].delta.content or ""
                            full_response += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        
                        # Save the complete response to conversation history
                        save_to_conversation_history({
                            "role": "assistant",
                            "content": full_response
                        })
                        
                    except Exception as e:
                        print(f"Groq API Error: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return Response(generate_response(), mimetype='text/event-stream')
                
            except Exception as e:
                print(f"General Error: {str(e)}")
                return jsonify({
                    "status": "error",
                    "error": "An unexpected error occurred",
                    "details": str(e)
                }), 500
    
    # For GET requests, render the template with chat history
    chat_history = load_conversation_history()
    return render_template("chatbot.html", chat_history=chat_history)

def get_user_upload_directory(username):
    user_dir = os.path.join(UPLOAD_FOLDER, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

def extract_page_text(pdf_path, page_number):
    try:
        reader = PdfReader(pdf_path)
        if page_number < 1 or page_number > len(reader.pages):
            return f"Error: Page number must be between 1 and {len(reader.pages)}"
        
        page = reader.pages[page_number - 1]  # Convert to 0-based index
        return page.extract_text()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Store the filepath in session for later use
            session['current_pdf'] = filepath
            
            # Get total pages
            pdf_reader = PdfReader(filepath)
            total_pages = len(pdf_reader.pages)
            
            return jsonify({
                'status': 'success',
                'total_pages': total_pages
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        page_number = int(data.get('page_number', 1))
        
        filepath = session.get('current_pdf')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Please upload a PDF first'}), 400
        
        # Extract text from the specified page
        pdf_reader = PdfReader(filepath)
        if page_number < 1 or page_number > len(pdf_reader.pages):
            return jsonify({'error': 'Invalid page number'}), 400
        
        page = pdf_reader.pages[page_number - 1]
        extracted_text = page.extract_text()
        
        # Generate questions
        question_maker = QuestionMaker()
        questions = question_maker.generate_questions(extracted_text)
        
        if questions:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Failed to generate questions'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/questions')
def questions():
    return render_template('questions.html')

# Add this route to get the generated questions
@app.route('/get_questions')
def get_questions():
    try:
        with open('questions/generated_questions.json', 'r') as f:
            questions = json.load(f)
        return jsonify(questions)
    except FileNotFoundError:
        return jsonify([])

# Add these routes to handle PDF upload and chat
@app.route('/upload_chat_pdf', methods=['POST'])
def upload_chat_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            chatbot = Chatbot()
            if chatbot.process_pdf(filepath):
                session['pdf_chat_enabled'] = True
                session['current_pdf'] = filename
                total_pages = chatbot.get_total_pages()
                return jsonify({
                    'status': 'success',
                    'message': 'File uploaded successfully',
                    'total_pages': total_pages
                })
            else:
                return jsonify({'error': 'Failed to process PDF'}), 500
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat')
def get_chat_response():
    user_message = request.args.get('message', '')
    page_number = request.args.get('page', type=int)
    
    if not user_message:
        return 'No message provided', 400

    chatbot = Chatbot()
    if session.get('pdf_chat_enabled'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['current_pdf'])
        chatbot.process_pdf(filepath)

    def generate():
        for token in chatbot.get_response(user_message, page_number):
            yield f"data: {json.dumps({'content': token})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

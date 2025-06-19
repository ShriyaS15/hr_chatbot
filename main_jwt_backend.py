# GeoBot Backend: JWT Auth + Excel + Leave Policy Chatbot
from flask import Flask, request, jsonify, send_from_directory
import datetime
import os
import json
import pandas as pd
import numpy as np
import fitz
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from functools import wraps
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import random
import jwt as pyjwt
import jwt
from docx import Document

# --- Init Flask ---
app = Flask(__name__)
app.secret_key = 'super_secret_key'

# --- Dummy Users ---
users_db = {
    "employee1": {"password": "pass123", "role": "employee"},
    "hradmin": {"password": "admin456", "role": "hr"}
}

# --- JWT Decorator ---


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            return f(data, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 403
        except Exception as e:
            return jsonify({'error': str(e)}), 403
    return decorated


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username or password missing'}), 400

        user = users_db.get(username)
        if user and user['password'] == password:
            token = pyjwt.encode({
                'username': username,
                'role': user['role'],
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)
            }, app.secret_key, algorithm='HS256')

            if isinstance(token, bytes):
                token = token.decode('utf-8')

            return jsonify({'token': token})

        return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        print(f"\U0001f525 Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# --- Model and Manager ---


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class LeaveManager:
    def __init__(self, excel_path):
        try:
            self.df = pd.read_excel(excel_path, skiprows=3)
            self.df.columns = self.df.columns.str.strip()
        except Exception as e:
            print(f"⚠️ Error loading Excel: {e}. Creating dummy data")
            self.df = pd.DataFrame({
                'Name': ['John Doe', 'Jane Smith'],
                'Leave Type': ['Casual', 'Sick'],
                'Transaction Type': ['Taken', 'Taken'],
                'Days': [5, 3],
                'Employee ID': ['E001', 'E002'],
                'Department': ['HR', 'IT']
            })

    def get_user_data(self, name):
        return self.df[self.df['Name'].str.lower() == name.lower()]

    def total_days_by_type(self, name, leave_type, txn_type="Taken"):
        df = self.get_user_data(name)
        df = df[(df['Leave Type'].str.lower() == leave_type.lower())
                & (df['Transaction Type'] == txn_type)]
        return df['Days'].sum()

    def all_leave_summary(self, name):
        df = self.get_user_data(name)
        summary = df[df['Transaction Type'] == 'Taken'].groupby('Leave Type')[
            'Days'].sum()
        return summary.to_dict()


class ChatbotAssistant:
    def __init__(self, intents_path, model_path, policy_path, excel_path):
        self.intents_path = intents_path
        self.model_path = model_path
        self.policy_path = policy_path
        try:
            self.leave_mgr = LeaveManager(excel_path)
        except:
            print("⚠️ Leave manager initialization failed, using dummy data")
            self.leave_mgr = LeaveManager("")
        self.responses = {}
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.model = None
        self.current_user = None
        self.policy_text = ""
        self.load_policy_text()
        self.build_model()

    def update_policy_document(self, file_path):
        try:
            doc = fitz.open(file_path)
            text = []
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    if isinstance(block, (list, tuple)) and len(block) > 4:
                        text.append(block[4].strip())
            self.policy_text = " ".join(text).lower()
            print(f"✅ Policy document updated from: {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to update policy document: {e}")

    def tokenize_and_lemmatize(self, text):
        try:
            lemmatizer = nltk.WordNetLemmatizer()
            return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]
        except:
            return text.lower().split()

    def bag_of_words(self, words):
        return [1 if w in words else 0 for w in self.vocabulary]

    def parse_intents(self):
        try:
            with open(self.intents_path, 'r') as f:
                data = json.load(f)
            for intent in data['intents']:
                tag = intent['tag']
                self.intents.append(tag)
                self.responses[tag] = intent['responses']
                for pattern in intent['patterns']:
                    words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(words)
                    self.documents.append((words, tag))
            self.vocabulary = sorted(list(set(self.vocabulary)))
            self.intents = sorted(list(set(self.intents)))
        except Exception as e:
            print(f"Error parsing intents: {e}")
            self.intents = ['greeting', 'goodbye']
            self.vocabulary = ['hello', 'hi', 'bye', 'goodbye']
            self.responses = {
                'greeting': ['Hello! How can I help you?'],
                'goodbye': ['Goodbye! Have a great day!']
            }

    def prepare_data(self):
        X, y = [], []
        for doc, intent in self.documents:
            bow = self.bag_of_words(doc)
            X.append(bow)
            y.append(self.intents.index(intent))
        if not X:
            X = [[1, 0, 0, 0], [0, 0, 1, 1]]
            y = [0, 1]
        return np.array(X), np.array(y)

    def load_policy_text(self):
        try:
            doc = fitz.open(self.policy_path)
            text = []
            for page in doc:
                blocks = page.get_text("blocks")  # preserve structure
                for block in blocks:
                    if isinstance(block, (list, tuple)) and len(block) > 4:
                        text.append(block[4].strip())
            self.policy_text = " ".join(text).lower()
            print(
                f"✅ Policy PDF loaded. Length: {len(self.policy_text)} chars")
        except Exception as e:
            print(f"⚠️ Policy PDF not found or unreadable: {e}")
            self.policy_text = "company leave policy: employees get 20 vacation days, 10 sick days per year."

    def train_model(self):
        X, y = self.prepare_data()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        self.model = ChatbotModel(X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(1000):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        print("✅ Model training completed")

    def build_model(self):
        self.parse_intents()
        if os.path.exists(self.model_path):
            try:
                X, y = self.prepare_data()
                self.model = ChatbotModel(X.shape[1], len(self.intents))
                self.model.load_state_dict(torch.load(
                    self.model_path, weights_only=True))
                print("✅ Loaded pretrained model.")
            except Exception as e:
                print("⚠️ Error loading model, retraining:", e)
                self.train_model()
                torch.save(self.model.state_dict(), self.model_path)
        else:
            self.train_model()
            torch.save(self.model.state_dict(), self.model_path)

    def process_message(self, msg):
        msg = msg.lower()
        if "my name is" in msg:
            name = msg.split("my name is")[-1].strip().title()
            if not self.leave_mgr.get_user_data(name).empty:
                self.current_user = name
                return f"Hi {name}, your record has been loaded. How can I help you?"
            return "I couldn't find your name in the system. Please check the spelling or contact HR."

        if any(keyword in msg for keyword in ["leave policy", "lop", "casual leave", "annual leave", "sick leave"]):
            return self.search_policy(msg)

        if self.current_user:
            if "how many" in msg or "taken" in msg:
                for lt in ["casual", "sick", "earned", "paid", "lop"]:
                    if lt in msg:
                        days = self.leave_mgr.total_days_by_type(
                            self.current_user, lt)
                        return f"You have taken {days:.1f} {lt.title()} Leave(s) so far."
            if "leave summary" in msg or "total leaves" in msg:
                summary = self.leave_mgr.all_leave_summary(self.current_user)
                if summary:
                    summary_text = "Your leave summary:\n"
                    for leave_type, days in summary.items():
                        summary_text += f"- {leave_type}: {days} days\n"

                    return summary_text
        try:
            tokens = self.tokenize_and_lemmatize(msg)
            bow = self.bag_of_words(tokens)
            X = torch.tensor([bow], dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                out = self.model(X)
                tag_idx = torch.argmax(out, dim=1).item()
                tag = self.intents[tag_idx]
            return random.choice(self.responses.get(tag, ["Sorry, I didn't understand that. Can you please rephrase?"]))
        except Exception as e:
            print(f"Error in message processing: {e}")
            return "I'm having trouble processing your request. Please try again or contact HR for assistance."


# Initialize assistant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
assistant = ChatbotAssistant(
    intents_path=os.path.join(BASE_DIR, 'intents.json'),
    model_path=os.path.join(BASE_DIR, 'chatbot_model.pth'),
    excel_path=os.path.join(BASE_DIR, 'leave_data.xlsx'),
    policy_path=os.path.join(BASE_DIR, 'leave_policy.pdf')
)


@app.route('/')
def index():
    return send_from_directory('.', 'main_jwt_backend.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


@app.route('/chat', methods=['POST'])
@token_required
def chat(payload):
    try:
        msg = request.json.get('message', '')
        reply = assistant.process_message(msg)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"\U0001f525 Chat error: {e}")
        return jsonify({"reply": "Internal server error occurred."}), 500


@app.route('/admin/dashboard', methods=['GET'])
@token_required
def admin_dashboard(payload):
    if payload.get('role') != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    try:
        dataset_info = {
            'total_records': len(assistant.leave_mgr.df) if hasattr(assistant.leave_mgr, 'df') else 0,
            'columns': list(assistant.leave_mgr.df.columns) if hasattr(assistant.leave_mgr, 'df') else [],
            'sample_data': assistant.leave_mgr.df.head(3).to_dict('records') if hasattr(assistant.leave_mgr, 'df') else [],
            'policy_loaded': len(assistant.policy_text) > 100,
            'policy_length': len(assistant.policy_text)
        }
        return jsonify(dataset_info)
    except Exception as e:
        print(f"Dashboard error: {e}")
        return jsonify({'error': f'Dashboard error: {str(e)}'}), 500


@app.route('/admin/upload', methods=['POST'])
@token_required
def upload_document(payload):
    if payload.get('role') != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        filename = file.filename
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        ext = filename.lower().split('.')[-1]
        if ext in ['xlsx', 'xls', 'csv']:
            assistant.leave_mgr = LeaveManager(file_path)
            return jsonify({'status': f'"{filename}" uploaded and processed as spreadsheet.'})
        elif ext in ['pdf', 'docx', 'doc', 'txt']:
            assistant.update_policy_document(file_path)
            return jsonify({'status': f'"{filename}" uploaded and used as policy document.'})
        else:
            return jsonify({'status': f'"{filename}" uploaded but not processed (unknown format).'})
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory('uploads', filename)


@app.route('/admin/train', methods=['POST'])
@token_required
def retrain_model(payload):
    if payload.get('role') != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    try:
        assistant.train_model()
        torch.save(assistant.model.state_dict(), assistant.model_path)
        return jsonify({'status': '✅ Model retrained successfully'})
    except Exception as e:
        print(f"Model retraining error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
    except:
        print("⚠️ NLTK download failed, using basic tokenization")
    app.run(debug=True)

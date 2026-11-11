import uuid
import json
import os
import shutil
import tempfile
from flask import Flask, request, jsonify, render_template_string, session
# Import the AI bot class
from import_re import EnhancedLearningQABot
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'AIzaSyB02IO2JL8k1je861YYoE0ujVZQ_TfFeiQ'  # Replace with a secure random key in production

# Instantiate the bot once
bot = EnhancedLearningQABot()

# --- User-specific persistent histories ---
# Vercel's app directory is read-only; only /tmp is writable there.
if os.environ.get('VERCEL'):
    USER_HISTORY_FILE = os.path.join(tempfile.gettempdir(), 'user_histories.json')
    if not os.path.exists(USER_HISTORY_FILE) and os.path.exists('user_histories.json'):
        shutil.copyfile('user_histories.json', USER_HISTORY_FILE)
else:
    USER_HISTORY_FILE = 'user_histories.json'
try:
    with open(USER_HISTORY_FILE, 'r') as f:
        all_user_histories = json.load(f)
except Exception:
    all_user_histories = {}

def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def save_all_histories():
    with open(USER_HISTORY_FILE, 'w') as f:
        json.dump(all_user_histories, f)

def get_latest_session(user_sessions):
    if not user_sessions:
        return None
    return max(user_sessions, key=lambda s: s['created_at'])

# --- SESSION ENDPOINTS ---
@app.route('/start_session', methods=['POST'])
def start_session():
    user_id = get_user_id()
    session_id = str(uuid.uuid4())
    session_entry = {
        'session_id': session_id,
        'created_at': datetime.utcnow().isoformat(),
        'messages': []
    }
    user_sessions = all_user_histories.get(user_id, [])
    user_sessions.append(session_entry)
    all_user_histories[user_id] = user_sessions
    save_all_histories()
    return jsonify({'session_id': session_id, 'created_at': session_entry['created_at']})

@app.route('/sessions', methods=['GET'])
def list_sessions():
    user_id = get_user_id()
    user_sessions = all_user_histories.get(user_id, [])
    return jsonify({'sessions': [
        {'session_id': s['session_id'], 'created_at': s['created_at'], 'num_messages': len(s['messages'])}
        for s in user_sessions
    ]})

@app.route('/session_history/<session_id>', methods=['GET'])
def get_session_history(session_id):
    user_id = get_user_id()
    user_sessions = all_user_histories.get(user_id, [])
    for s in user_sessions:
        if s['session_id'] == session_id:
            return jsonify({'messages': s['messages']})
    return jsonify({'error': 'Session not found'}), 404

# --- MODIFIED /history ENDPOINT ---
@app.route('/history', methods=['GET'])
def get_history():
    user_id = get_user_id()
    session_id = request.args.get('session_id')
    user_sessions = all_user_histories.get(user_id, [])
    session_obj = None
    if session_id:
        for s in user_sessions:
            if s['session_id'] == session_id:
                session_obj = s
                break
    else:
        session_obj = get_latest_session(user_sessions)
    history = []
    if session_obj:
        for entry in session_obj['messages']:
            if 'question' in entry:
                history.append({'role': 'user', 'text': entry['question']})
            if 'response' in entry:
                history.append({'role': 'ai', 'text': entry['response']})
    return jsonify({'history': history, 'session_id': session_obj['session_id'] if session_obj else None})

# --- MODIFIED /predict ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    user_id = get_user_id()
    data = request.get_json()
    user_input = data.get('input') if data else None
    session_id = data.get('session_id') if data else None
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    user_sessions = all_user_histories.get(user_id, [])
    session_obj = None
    if session_id:
        for s in user_sessions:
            if s['session_id'] == session_id:
                session_obj = s
                break
    if not session_obj:
        # If no session_id provided or not found, create a new session
        session_id = str(uuid.uuid4())
        session_obj = {
            'session_id': session_id,
            'created_at': datetime.utcnow().isoformat(),
            'messages': []
        }
        user_sessions.append(session_obj)
        all_user_histories[user_id] = user_sessions
    # Append user message
    session_obj['messages'].append({'question': user_input})
    # Get AI response
    try:
        response = bot.ask_gemini(user_input)
    except Exception as e:
        return jsonify({"message": f"Backend error: {e}", "session_id": session_id}), 500
    # Append AI message
    session_obj['messages'][-1]['response'] = response
    save_all_histories()
    # Also append to bot's persistent memory (optional: for global learning)
    bot.conversation_history.append({'question': user_input, 'response': response})
    bot.save_ml_data()
    return jsonify({"message": response, "session_id": session_id})

# --- MODIFIED /clear_history ENDPOINT ---
@app.route('/clear_history', methods=['POST'])
def clear_history():
    user_id = get_user_id()
    # Clear all sessions for this user
    all_user_histories[user_id] = []
    save_all_histories()
    session['history'] = []
    return jsonify({'success': True})

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liam AI Web UI</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
            h1 {
                text-align: center;
                color: #2d3a4b;
                margin-top: 40px;
                margin-bottom: 30px;
                letter-spacing: 1px;
            }
            #chat-container {
                background: #fff;
                max-width: 480px;
                margin: 0 auto;
                padding: 24px 18px 16px 18px;
                border-radius: 18px;
                box-shadow: 0 4px 24px rgba(44, 62, 80, 0.08);
                display: flex;
                flex-direction: column;
                gap: 18px;
                min-height: 60vh;
                height: 60vh;
            }
            #session-controls {
                display: flex;
                gap: 10px;
                align-items: center;
                margin-bottom: 10px;
            }
            #session-select {
                flex: 1;
                padding: 8px 10px;
                border-radius: 8px;
                border: 1px solid #cfd8dc;
                font-size: 1em;
            }
            #new-session-btn {
                background: #4caf50;
                color: #fff;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s, transform 0.1s;
            }
            #new-session-btn:hover {
                background: #388e3c;
                transform: translateY(-2px) scale(1.03);
            }
            #chat-history {
                flex: 1;
                overflow-y: auto;
                padding-bottom: 8px;
                margin-bottom: 10px;
            }
            .message {
                display: flex;
                margin-bottom: 10px;
            }
            .message.user {
                justify-content: flex-end;
            }
            .message.ai {
                justify-content: flex-start;
            }
            .bubble {
                max-width: 75%;
                padding: 12px 16px;
                border-radius: 16px;
                font-size: 1.08em;
                word-break: break-word;
                white-space: pre-line;
            }
            .bubble.user {
                background: linear-gradient(90deg, #5b9df9 0%, #3a8dde 100%);
                color: #fff;
                border-bottom-right-radius: 4px;
            }
            .bubble.ai {
                background: #f1f5fa;
                color: #2d3a4b;
                border-bottom-left-radius: 4px;
            }
            #ai-form {
                display: flex;
                gap: 10px;
                margin-top: 0;
            }
            #user-input {
                flex: 1;
                padding: 12px 14px;
                border: 1px solid #cfd8dc;
                border-radius: 8px;
                font-size: 1.1em;
                outline: none;
                transition: border 0.2s;
            }
            #user-input:focus {
                border: 1.5px solid #5b9df9;
            }
            button {
                background: linear-gradient(90deg, #5b9df9 0%, #3a8dde 100%);
                color: #fff;
                border: none;
                border-radius: 8px;
                padding: 0 22px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(44, 62, 80, 0.07);
                transition: background 0.2s, transform 0.1s;
            }
            button:hover {
                background: linear-gradient(90deg, #3a8dde 0%, #5b9df9 100%);
                transform: translateY(-2px) scale(1.03);
            }
            #clear-history-btn {
                background: #f44336;
                color: #fff;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                margin-bottom: 10px;
                margin-left: auto;
                margin-right: auto;
                display: block;
                box-shadow: 0 2px 8px rgba(44, 62, 80, 0.07);
                transition: background 0.2s, transform 0.1s;
            }
            #clear-history-btn:hover {
                background: #d32f2f;
                transform: translateY(-2px) scale(1.03);
            }
            @media (max-width: 600px) {
                #chat-container {
                    max-width: 98vw;
                    padding: 10px 2vw;
                }
                h1 {
                    font-size: 1.3em;
                }
            }
        </style>
    </head>
    <body>
        <h1>Liam AI Web Interface</h1>
        <button id="clear-history-btn">Clear History</button>
        <div id="chat-container">
            <div id="session-controls">
                <select id="session-select"></select>
                <button id="new-session-btn">New Session</button>
            </div>
            <div id="chat-history"></div>
            <form id="ai-form" autocomplete="off">
                <input type="text" id="user-input" placeholder="Type your message..." required autofocus autocomplete="off">
                <button type="submit">Send</button>
            </form>
        </div>
        <script>
        const chatHistory = document.getElementById('chat-history');
        const aiForm = document.getElementById('ai-form');
        const userInput = document.getElementById('user-input');
        const clearBtn = document.getElementById('clear-history-btn');
        const sessionSelect = document.getElementById('session-select');
        const newSessionBtn = document.getElementById('new-session-btn');

        let currentSessionId = null;

        // Fetch and render all sessions for the dropdown
        async function loadSessions(selectLatest = false) {
            const res = await fetch('/sessions');
            const data = await res.json();
            sessionSelect.innerHTML = '';
            if (data.sessions && data.sessions.length > 0) {
                data.sessions.forEach((s, idx) => {
                    const opt = document.createElement('option');
                    opt.value = s.session_id;
                    opt.text = `Session ${idx+1} (${new Date(s.created_at).toLocaleString()})`;
                    sessionSelect.appendChild(opt);
                });
                if (selectLatest) {
                    currentSessionId = data.sessions[data.sessions.length-1].session_id;
                    sessionSelect.value = currentSessionId;
                } else if (currentSessionId) {
                    sessionSelect.value = currentSessionId;
                } else {
                    currentSessionId = data.sessions[0].session_id;
                    sessionSelect.value = currentSessionId;
                }
            } else {
                // No sessions, create one automatically
                await startNewSession();
                return;
            }
            await loadHistory();
        }

        // Start a new session
        async function startNewSession() {
            const res = await fetch('/start_session', {method: 'POST'});
            const data = await res.json();
            currentSessionId = data.session_id;
            await loadSessions(true);
        }

        // Load chat history for the current session
        async function loadHistory() {
            if (!currentSessionId) return;
            const res = await fetch(`/history?session_id=${currentSessionId}`);
            const data = await res.json();
            chatHistory.innerHTML = '';
            if (Array.isArray(data.history)) {
                data.history.forEach(msg => {
                    appendMessage(msg.role, msg.text);
                });
            }
        }

        // Append a message to the chat
        function appendMessage(role, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + role;
            const bubble = document.createElement('div');
            bubble.className = 'bubble ' + role;
            bubble.innerText = text;
            msgDiv.appendChild(bubble);
            chatHistory.appendChild(msgDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        // Handle session selection change
        sessionSelect.onchange = async function() {
            currentSessionId = sessionSelect.value;
            await loadHistory();
        };

        // Handle new session button
        newSessionBtn.onclick = async function() {
            await startNewSession();
        };

        // Handle form submit (send message)
        aiForm.onsubmit = async function(e) {
            e.preventDefault();
            const input = userInput.value.trim();
            if (!input || !currentSessionId) return;
            appendMessage('user', input);
            userInput.value = '';
            userInput.focus();
            // Show a loading bubble
            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'message ai';
            const loadingBubble = document.createElement('div');
            loadingBubble.className = 'bubble ai';
            loadingBubble.innerText = '...';
            loadingMsg.appendChild(loadingBubble);
            chatHistory.appendChild(loadingMsg);
            chatHistory.scrollTop = chatHistory.scrollHeight;
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: input, session_id: currentSessionId, use_gemini: true})
                });
                const data = await response.json();
                chatHistory.removeChild(loadingMsg);
                appendMessage('ai', data.message || JSON.stringify(data));
                if (data.session_id && data.session_id !== currentSessionId) {
                    // If a new session was created, reload sessions
                    currentSessionId = data.session_id;
                    await loadSessions();
                }
            } catch (err) {
                chatHistory.removeChild(loadingMsg);
                appendMessage('ai', 'Error: ' + err.message);
            }
        };

        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                aiForm.dispatchEvent(new Event('submit'));
                e.preventDefault();
            }
        });

        clearBtn.onclick = async function() {
            if (!confirm('Are you sure you want to clear the entire chat history?')) return;
            await fetch('/clear_history', {method: 'POST'});
            await loadSessions();
        };

        // Initial load
        loadSessions();
        </script>
    </body>
    </html>
    """)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
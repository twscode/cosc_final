import os
import re
import asyncio
import pandas as pd
import openai
from flask import Flask, request, jsonify, abort, session
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4

app = Flask(__name__)
app.secret_key = '3fa85f64-5717-4562-b3fc-2c963f66afa6'

data_path = '/mnt/data/course_list.csv'
try:
    assignments = pd.read_csv(data_path)
except Exception as e:
    app.logger.error("Failed to load data: %s", str(e))

handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
openai.api_key = os.getenv('sk-FAKEo7JFKAjD5BvIY8CzCu1e4wE82FSdn08Xn9lo')

@app.route('/chat', methods=['POST'])
def chat():
    session_id = session.get('session_id', str(uuid4()))
    session['session_id'] = session_id

    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        logger.error("No message provided in request")
        abort(400, description="No message provided")

    try:
        response = asyncio.run(process_message(user_message, session_id))
        return jsonify({'response': response}), 200
    except Exception as e:
        logger.error("Error in processing message: %s", str(e))
        abort(500, description="Internal server error")

async def process_message(message, session_id):
    intent = await detect_intent(message)
    handler = intent_mapping().get(intent, fetch_openai_response)
    return await handler(message, session_id)

def intent_mapping():
    return {
        'course_info': handle_course_info,
        'faculty_info': handle_faculty_info,
        'enrollment': handle_enrollment,
        'academic_support': handle_academic_support,
        'career_opportunities': handle_career_opportunities,
        'accreditation': handle_accreditation,
        'general_info': handle_general_info,
        'feedback': handle_feedback
    }

async def detect_intent(message):
    training_data = {
        'course_info': ["Can you provide information about the Introduction to Computer Science course?",
                        "Tell me about the prerequisites for the Artificial Intelligence course.",
                        "What are the elective options for the Computer Science major?"],
        'faculty_info': ["Who teaches the Artificial Intelligence course this semester?",
                         "Can I find Dr. Wang's office hours?",
                         "Are there any female professors in the Computer Science department?"],
        'enrollment': ["How do I register for the Game Design course?",
                       "When is the deadline for adding courses this semester?",
                       "Where can I find information about dropping a class?"],
        'academic_support': ["I'm struggling with my programming assignment. Can you help?",
                             "How can I schedule a meeting with my academic advisor?",
                             "Do you have resources for studying algorithms?"],
        'career_opportunities': ["What internship opportunities are available for Computer Science majors?",
                                 "Can you recommend any networking events for students interested in software development?",
                                 "Where can I find resources for preparing my resume for tech internships?"],
        'accreditation': ["Are the chemistry programs at SCMNS accredited?",
                          "Is the medical laboratory science program certified?",
                          "Can you confirm the accreditation status of SCMNS programs?"],
        'general_info': ["What are the office hours for the Computer Science department?",
                         "How do I access the computer labs on campus?",
                         "Where can I find information about scholarships for Computer Science students?"],
        'feedback': ["Was this response helpful? Please rate it with a thumbs up or thumbs down.",
                     "Do you have any suggestions to improve the chatbot's performance?"]
    }
    documents, intent_labels = [], []
    for intent, samples in training_data.items():
        documents.extend(samples)
        intent_labels.extend([intent] * len(samples))

    vectorizer = TfidfVectorizer().fit(documents)
    message_vector = vectorizer.transform([message])
    document_vectors = vectorizer.transform(documents)
    similarities = cosine_similarity(message_vector, document_vectors)
    if similarities.max() > 0.3:
        return intent_labels[similarities.argmax()]
    return 'general'

async def handle_course_info(message, session_id):
    return f"Session {session_id}: Detailed course information response"

async def handle_faculty_info(message, session_id):
    return f"Session {session_id}: Faculty information response"

async def handle_enrollment(message, session_id):
    return f"Session {session_id}: Enrollment information response"

async def handle_academic_support(message, session_id):
    return f"Session {session_id}: Academic support information"

async def handle_career_opportunities(message, session_id):
    return f"Session {session_id}: Career opportunities information"

async def handle_accreditation(message, session_id):
    return f"Session {session_id}: Accreditation information response"

async def handle_general_info(message, session_id):
    return f"Session {session_id}: General information response"

async def handle_feedback(message, session_id):
    return f"Session {session_id}: Feedback processing response"

async def fetch_openai_response(message, session_id):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message['content']

if __name__ == '__main__':
    app.run(port=5000, debug=True)

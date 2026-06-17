from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.prompt import *
import os
import logging
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("medical-chatbot")

app = Flask(__name__)

# Native CORS implementation
@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logger.info("Downloading Hugging Face embeddings...")
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

logger.info("Connecting to existing Pinecone index...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Use OpenRouter API with a verified working vision model
logger.info("Initializing LLM client (openai/gpt-4o-mini)...")
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.4,
    max_tokens=500,
    openai_api_base="https://openrouter.ai/api/v1",
)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "medical-chatbot-api",
        "pinecone_index": index_name
    })


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        image_file = None
        if request.method == "GET":
            msg = request.args.get("msg")
        else:
            # POST request
            if request.is_json:
                data = request.get_json()
                msg = data.get("msg")
            else:
                msg = request.form.get("msg")
                image_file = request.files.get("image")

        if not msg and not image_file:
            logger.warning("Empty or missing message and image received")
            return jsonify({
                "status": "error",
                "message": "Either message text or an image is required."
            }), 400

        logger.info(f"Received query: '{msg[:100] if msg else '[Image only]'}'")
        
        # 1. RAG context retrieval
        context = ""
        if msg:
            logger.info("Retrieving context from Pinecone...")
            retrieved_docs = retriever.invoke(msg)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
        else:
            logger.info("Image-only query, skipping RAG context retrieval...")

        # 2. Build system prompt message
        formatted_system_prompt = system_prompt.format(context=context)
        messages = [
            SystemMessage(content=formatted_system_prompt)
        ]

        # 3. Handle base64 image parsing if present
        if image_file:
            logger.info("Reading and encoding attached image file...")
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            mime_type = image_file.content_type or 'image/jpeg'
            
            human_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
            
            # If text prompt is also present, append it as a text block
            if msg:
                human_content.insert(0, {
                    "type": "text",
                    "text": msg
                })
                
            messages.append(HumanMessage(content=human_content))
        else:
            # Text-only human message
            messages.append(HumanMessage(content=msg))

        # 4. Invoke LLM client
        logger.info("Invoking LLM model...")
        response = llm.invoke(messages)
        answer = response.content
        
        if not answer:
            logger.error("LLM returned an empty answer")
            return jsonify({
                "status": "error",
                "message": "Failed to generate an answer."
            }), 500

        logger.info("Successfully generated AI answer")
        return jsonify({
            "status": "success",
            "answer": answer
        })

    except Exception as e:
        logger.exception(f"Unhandled exception in /get route: {e}")
        return jsonify({
            "status": "error",
            "message": "Sorry, I encountered an error processing your request. Please try again."
        }), 500


if __name__ == '__main__':
    logger.info("Starting Medical Chatbot on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=True)

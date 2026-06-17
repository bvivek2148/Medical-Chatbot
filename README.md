# 🩺 Medical Chatbot

A lightweight, intelligent medical chatbot that uses OpenAI GPT models with vector embeddings from Pinecone to respond to health-related queries.

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bvivek2148/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2️⃣ Create a Conda Environment

```bash
conda create -n medichat python=3.12 -y
conda activate medichat
```

### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file in the root directory with the following:

```ini
PINECONE_API_KEY="your_pinecone_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

### 5️⃣ Index Medical Documents

```bash
python store_index.py
```

### 6️⃣ Run the Application

```bash
python app.py
```

Visit: [http://localhost:8080](http://localhost:8080)

---

## 🧠 Tech Stack

- Python
- Flask
- LangChain
- OpenAI GPT
- Pinecone

---

# 🛠️ AWS CI/CD Deployment with GitHub Actions

### 1. AWS Setup

#### ✅ IAM User with the following permissions:
- `AmazonEC2FullAccess`
- `AmazonEC2ContainerRegistryFullAccess`

#### ✅ Services Used:
- **EC2**: For hosting the application
- **ECR**: For storing Docker images

### 2. ECR Repository

Create an ECR repo (e.g., `medicalchatbot`)  
Example URI:  
`970547337635.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot`

### 3. EC2 Instance

- Launch an Ubuntu EC2 instance
- Install Docker:

```bash
sudo apt-get update && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 4. Self-hosted GitHub Runner

- Go to: **GitHub > Repo Settings > Actions > Runners**
- Add a new Ubuntu runner and follow the setup instructions

### 5. GitHub Secrets Required

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REPO`
- `PINECONE_API_KEY`
- `OPENAI_API_KEY`

---

### ✅ Happy Building! 💬🧠

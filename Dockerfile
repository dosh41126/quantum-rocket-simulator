FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-tk \
    libgl1-mesa-glx \
    curl \
    iptables \
    dnsutils \
    openssl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install NLTK
RUN pip install --no-cache-dir nltk==3.8.1

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Save the export script into the container
RUN echo '#!/usr/bin/env python3\n\
import os, json, weaviate\n\
from weaviate.embedded import EmbeddedOptions\n\
PERSISTED_DB_PATH = os.path.expanduser("~/humoid_data/weaviate")\n\
OUTPUT_FILE = "weaviate_history.txt"\n\
def fetch_all_interactions(client, limit=100):\n\
    try:\n\
        query = {"query": f"""{{ Get {{ InteractionHistory(limit: {limit}) {{ user_id response response_time }} }} }}"""}\n\
        response = client.query.raw(json.dumps(query))\n\
        data = response.get("data", {}).get("Get", {}).get("InteractionHistory", [])\n\
        return data\n\
    except Exception as e:\n\
        print(f"❌ Error fetching from Weaviate: {e}")\n\
        return []\n\
def save_to_txt(interactions, filename):\n\
    try:\n\
        with open(filename, "w", encoding="utf-8") as f:\n\
            for item in interactions:\n\
                role = "User" if "bot" not in item["user_id"].lower() else "Bot"\n\
                f.write(f"[{role}] ({item["response_time"]}):\\n{item["response"]}\\n\\n")\n\
        print(f"✅ Exported {len(interactions)} interactions to {filename}")\n\
    except Exception as e:\n\
        print(f"❌ Error writing to file: {e}")\n\
if __name__ == "__main__":\n\
    if not os.path.exists(PERSISTED_DB_PATH):\n\
        print(f"❌ Local Weaviate DB path not found: {PERSISTED_DB_PATH}")\n\
        exit(1)\n\
    client = weaviate.Client(embedded_options=EmbeddedOptions(persisted_path=PERSISTED_DB_PATH))\n\
    interactions = fetch_all_interactions(client, limit=500)\n\
    save_to_txt(interactions, OUTPUT_FILE)' > /app/export_weaviate_interactions.py

# Make vault passphrase env exporter
RUN openssl rand -hex 32 > /app/.vault_pass && \
    echo "export VAULT_PASSPHRASE=$(cat /app/.vault_pass)" > /app/set_env.sh && \
    chmod +x /app/set_env.sh

# Build and add secure startup script with limited IP access
RUN cat << 'EOF' > /app/firewall_start.sh
#!/bin/bash
set -e
source /app/set_env.sh

echo "[INFO] Setting up iptables firewall..."
iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

resolve_and_allow() {
  DOMAIN=$1
  echo "[INFO] Resolving $DOMAIN..."
  getent ahosts $DOMAIN | awk '/STREAM/ {print $1}' | sort -u | while read ip; do
    clean_ip=$(echo $ip | tr -d '"')
    if [[ $clean_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "[INFO] Allowing $clean_ip for $DOMAIN"
      iptables -A OUTPUT -d $clean_ip -j ACCEPT
    else
      echo "[WARN] Skipping invalid IP: $clean_ip"
    fi
  done
}

resolve_and_allow huggingface.co
resolve_and_allow objects.githubusercontent.com

iptables -A OUTPUT -j REJECT
echo "[INFO] Firewall active. Continuing..."

if [ ! -f /data/llama-2-7b-chat.ggmlv3.q8_0.bin ]; then
  echo "Downloading model file..."
  curl -L -o /data/llama-2-7b-chat.ggmlv3.q8_0.bin \
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --progress-bar
  echo "Verifying model file..."
  echo "3bfdde943555c78294626a6ccd40184162d066d39774bd2c98dae24943d32cc3  /data/llama-2-7b-chat.ggmlv3.q8_0.bin" | sha256sum -c -
else
  echo "Model file already exists, skipping download."
fi

ls -lh /data/llama-2-7b-chat.ggmlv3.q8_0.bin
export DISPLAY=:0
exec python main.py
EOF

# Make startup script executable
RUN chmod +x /app/firewall_start.sh

# Generate config.json
RUN python -c 'import random, string, json; print(json.dumps({ \
  "DB_NAME": "story_generator.db", \
  "WEAVIATE_ENDPOINT": "http://localhost:8079", \
  "WEAVIATE_QUERY_PATH": "/v1/graphql", \
  "LLAMA_MODEL_PATH": "/data/llama-2-7b-chat.ggmlv3.q8_0.bin", \
  "IMAGE_GENERATION_URL": "http://127.0.0.1:7860/sdapi/v1/txt2img", \
  "MAX_TOKENS": 3999, \
  "CHUNK_SIZE": 1250, \
  "API_KEY": "".join(random.choices(string.ascii_letters + string.digits, k=32)), \
  "WEAVIATE_API_URL": "http://localhost:8079/v1/objects", \
  "ELEVEN_LABS_KEY": "apikyhere" \
}))' > /app/config.json

CMD ["/app/firewall_start.sh"]

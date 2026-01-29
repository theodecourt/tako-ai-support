import requests
import json

# ====== CREDENCIAIS ======
INSTANCE_ID = "3EDF4DF2CBD0A27BEFA37E439F0C86D3"
INSTANCE_TOKEN = "A1B651F368B03E53D545DDC9"
CLIENT_TOKEN = "F70463fbad6d84b71b412f45221ea1ebfS"

PHONE = "5511976220012"  # sempre com DDI + DDD
MESSAGE = "🚀 Teste Z-API funcionando com client-token!"

# ====== ENDPOINT ======
url = f"https://api.z-api.io/instances/{INSTANCE_ID}/token/{INSTANCE_TOKEN}/send-text"

headers = {
    "Content-Type": "application/json",
    "client-token": CLIENT_TOKEN
}

payload = {
    "phone": PHONE,
    "message": MESSAGE
}

# ====== REQUEST ======
response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30
)

print("Status:", response.status_code)
print("Response:", response.text)

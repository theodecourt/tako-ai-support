# Tako Take-Home Exercise  
## Generative AI for Customer Support  

**Author:** Theo Decourt  

This repository contains the core code and prompting logic for a Generative AI–powered customer support system built as part of the **Tako AI Engineering Take-Home Exercise**.

The project focuses on **intent understanding, tone awareness, risk detection, and escalation decision-making** in a sensitive B2B payroll and HR context.

---

## 🔗 Project Links

### ▶️ Architecture Walkthrough Video (AWS)
https://youtu.be/ZVGCDaKzsq8  

⚠️ The AWS environment is private. This video visually demonstrates the full cloud infrastructure and execution flow.

---

### 💬 Live WhatsApp Demo
 https://bit.ly/tako-whatsapp-demo
 
**Best use case:**  
This chatbot is designed for B2B payroll and HR scenarios involving operational, legal, or financial risk.  
It performs best when asked concrete questions about payroll execution, terminations, compliance, deadlines, penalties, or error handling.

_The system is intentionally not optimized for small talk or open-ended conversation._

---

## 📌 About This Repository

This repository contains:

- The **AWS Lambda handler** responsible for orchestrating message processing  
- All **LLM prompts** 

⚠️ **Important:**  
This repository is **not meant to be run locally**.

The system relies on managed AWS services including:

- API Gateway  
- AWS Lambda  
- Amazon Bedrock (LLM + Agents)  
- DynamoDB (mutex + metadata storage)  
- Amazon S3 (knowledge base for RAG)  
- Z-API (WhatsApp integration)

As a result, the code here should be read as a **reference implementation**, not as a standalone application.

## 📄 Final Report

The complete technical report for this project is available below:

import json
import boto3
from pathlib import Path
import os
import http.client
import time
from botocore.exceptions import ClientError

# Bedrock client
bedrock = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-2"
)


# -------- Bedrock Flow invocation --------

FLOW_ID = "GDE6GAZUSZ"
FLOW_ALIAS_ID = "ZJ1SS18PPX"

def invoke_flow(full_prompt: str) -> dict:
    return bedrock.invoke_flow(
        flowIdentifier=FLOW_ID,
        flowAliasIdentifier=FLOW_ALIAS_ID,
        inputs=[
            {
                "nodeName": "FlowInputNode",
                "nodeOutputName": "document",
                "content": {
                    "document": full_prompt
                }
            }
        ]
    )


# -------- Bedrock salario atrasado agent invocation --------

AGENT_PAGAMENTO_ATRASADO_ID = "GUCCGBRVAH"
AGENT_PAGAMENTO_ATRASADO_ALIAS_ID = "S4BTLYMXTT"

def invoke_pagamento_atrasado_agent(user_message: str) -> dict:
    response = bedrock.invoke_agent(
        agentId=AGENT_PAGAMENTO_ATRASADO_ID,
        agentAliasId=AGENT_PAGAMENTO_ATRASADO_ALIAS_ID,
        sessionId="tako-session-1",
        inputText=user_message
    )

    final_text = ""

    # Agents retornam stream
    for event in response.get("completion", []):
        if "chunk" in event:
            final_text += event["chunk"]["bytes"].decode("utf-8")

    final_text = final_text.strip()

    # --- Tratativas simples ---
    if not final_text:
        return {
            "draft_answer": "Não foi possível gerar uma resposta automática para este caso no momento.",
            "confidence_score": 0.0
        }

    try:
        return json.loads(final_text)
    except json.JSONDecodeError:
        return {
            "draft_answer": "A resposta automática gerada não pôde ser interpretada corretamente.",
            "confidence_score": 0.0
        }


# -------- Bedrock demissao/recisao agent invocation --------

AGENT_DEMISSAO_RESCISAO_ID = "VPYOI3J5UU"
AGENT_DEMISSAO_RESCISAO_ALIAS_ID = "KTGEIAAQC3"

def invoke_demissao_recisao_agent(user_message: str) -> dict:
    response = bedrock.invoke_agent(
        agentId=AGENT_DEMISSAO_RESCISAO_ID,
        agentAliasId=AGENT_DEMISSAO_RESCISAO_ALIAS_ID,
        sessionId="tako-session-demissao-rescisao",
        inputText=user_message
    )

    final_text = ""

    # Agents retornam stream
    for event in response.get("completion", []):
        if "chunk" in event:
            final_text += event["chunk"]["bytes"].decode("utf-8")

    final_text = final_text.strip()

    if not final_text:
        return {
            "draft_answer": (
                "Não foi possível gerar uma resposta automática para essa solicitação "
                "no momento. A equipe da Tako irá analisar o caso."
            ),
            "confidence_score": 0.0
        }

    try:
        parsed = json.loads(final_text)

        # Blindagem extra: garante campos mínimos
        return {
            "draft_answer": parsed.get(
                "draft_answer",
                "Não foi possível gerar uma resposta automática completa para esse caso."
            ),
            "confidence_score": parsed.get("confidence_score", 0.0)
        }

    except json.JSONDecodeError:
        # Caso o agente responda texto puro
        return {
            "draft_answer": (
                "Recebemos sua solicitação sobre demissão ou rescisão. "
                "Ela exige uma análise mais cuidadosa e será encaminhada para revisão."
            ),
            "confidence_score": 0.0
        }


# -------- Prompt loading --------

def load_prompt(name: str) -> str:
    base_dir = Path(__file__).resolve().parent
    prompt_path = base_dir / "prompts" / f"{name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


# -------- Flow output extraction --------

def extract_flow_output(flow_response: dict) -> str:
    """
    Extracts the final output emitted by the FlowOutputNode.
    """
    for event in flow_response["responseStream"]:
        if "flowOutputEvent" in event:
            return event["flowOutputEvent"]["content"]["document"]

    raise RuntimeError("No flow output found")


# -------- Message analysis parsing --------

def parse_message_analysis(output_text: str) -> dict:
    parsed = json.loads(output_text)

    intencao = parsed.get("intencao")
    tom = parsed.get("tom")

    riscos = parsed.get("riscos", {})
    risco_legal = riscos.get("legal", False)
    risco_financeiro = riscos.get("financeiro", False)
    risco_emocional = riscos.get("emocional", False)

    mensagem_intermediaria = parsed.get("mensagem_intermediaria")

    return {
        "intencao": intencao,
        "tom": tom,
        "riscos": {
            "legal": risco_legal,
            "financeiro": risco_financeiro,
            "emocional": risco_emocional
        },
        "mensagem_intermediaria": mensagem_intermediaria
    }

# -------- Erro folha parsing --------

def parser_generico(output_text: str) -> dict:
    parsed = json.loads(output_text)

    draft_answer = parsed.get("draft_answer")
    confidence_score = parsed.get("confidence_score")

    return {
        "draft_answer": draft_answer,
        "confidence_score": confidence_score
    }

# -------- Intent-specific resolution agents --------

def handle_pagamento_atrasado(context: dict) -> dict:
    user_message = context["user_message"]

    agent_response = invoke_pagamento_atrasado_agent(user_message)

    return {
        "agent": "pagamento_atrasado_agent",
        "draft_answer": agent_response["draft_answer"],
        "confidence_score": agent_response["confidence_score"]
    }


def handle_erro_folha(context: dict) -> dict:

    base_prompt = load_prompt("erro_folha_agent")

    full_prompt = f"""{base_prompt}

    User input:
    \"\"\"
    {context["user_message"]}
    \"\"\"

    """

    flow_response = invoke_flow(full_prompt)
    agent_output = extract_flow_output(flow_response)
    parsed_output = parser_generico(agent_output)

    return {
        "agent": "erro_folha_agent",
        "draft_answer": parsed_output["draft_answer"],
        "confidence_score": parsed_output["confidence_score"]
    }


def handle_demissao_rescisao(context: dict) -> dict:

    user_message = context["user_message"]

    agent_output = invoke_demissao_recisao_agent(user_message)

    return {
        "agent": "demissao_rescisao_agent",
        "draft_answer": agent_output.get("draft_answer"),
        "confidence_score": agent_output.get("confidence_score")
    }



def handle_conformidade_legal(context: dict) -> dict:
    """
    Handles legal compliance questions using the general Bedrock Flow.
    """

    user_message = context.get("user_message")

    base_prompt = load_prompt("conformidade_legal_agent")

    full_prompt = f"""{base_prompt}

    User input:
    \"\"\"
    {user_message}
    \"\"\"
    """

    flow_response = invoke_flow(full_prompt)
    agent_output = extract_flow_output(flow_response)

    parsed_output = parser_generico(agent_output)

    return {
        "agent": "conformidade_legal_agent",
        "draft_answer": parsed_output["draft_answer"],
        "confidence_score": parsed_output["confidence_score"]
    }



def handle_duvida_geral(context: dict) -> dict:
    user_message = context.get("user_message")

    # Load duvida geral prompt
    base_prompt = load_prompt("duvida_geral_agent")

    full_prompt = f"""{base_prompt}

    User input:
    \"\"\"
    {user_message}
    \"\"\"
    """

    # Call Bedrock Flow
    flow_response = invoke_flow(full_prompt)
    agent_output = extract_flow_output(flow_response)

    # Parse agent output
    parsed_output = parser_generico(agent_output)

    return {
        "agent": "duvida_geral_agent",
        "draft_answer": parsed_output["draft_answer"],
        "confidence_score": parsed_output["confidence_score"]
    }


def handle_fallback(context: dict) -> dict:
    """
    Handles fallback cases (out-of-scope questions).
    """

    user_message = context.get("user_message", "")

    # Load fallback prompt
    base_prompt = load_prompt("fallback_agent")

    full_prompt = f"""{base_prompt}

    User input:
    \"\"\"
    {user_message}
    \"\"\"
    """

    # Invoke Bedrock flow
    flow_response = invoke_flow(full_prompt)

    # Extract raw model output (string)
    output_text = extract_flow_output(flow_response)

    # Parse JSON output
    parsed_output = parser_generico(output_text)

    return {
        "agent": "fallback_agent",
        "draft_answer": parsed_output["draft_answer"],
        "confidence_score": parsed_output["confidence_score"]
    }

# -------- Defines which intent-specific agent to invoke --------

def route_by_intent(intent: str, context: dict) -> dict:
    if intent == "pagamento_atrasado":
        return handle_pagamento_atrasado(context)

    if intent == "erro_folha":
        return handle_erro_folha(context)

    if intent == "demissao_rescisao":
        return handle_demissao_rescisao(context)

    if intent == "conformidade_legal":
        return handle_conformidade_legal(context)

    if intent == "duvida_geral":
        return handle_duvida_geral(context)

    return handle_fallback(context)

# -------- Define escalation --------

AUTO_SEND = "auto_send"
HUMAN_REVIEW = "human_review"
HUMAN_ONLY = "human_only"

def decide_escalation(tom: str, riscos: dict, confidence_score: float) -> str:
    risco_legal = riscos.get("legal", False)
    risco_financeiro = riscos.get("financeiro", False)
    risco_emocional = riscos.get("emocional", False)

    if confidence_score <= 0.6:
        return HUMAN_ONLY

    if (risco_legal or risco_financeiro) and risco_emocional and confidence_score < 0.8:
        return HUMAN_ONLY

    if confidence_score >= 0.85:
        return AUTO_SEND

    if confidence_score >= 0.75 and not risco_legal:
        return AUTO_SEND

    if confidence_score >= 0.65 and not any([risco_legal, risco_financeiro, risco_emocional]):
        return AUTO_SEND

    return HUMAN_REVIEW

# -------- Build resoponse generator prompt --------

def build_rewrite_prompt(
    user_message: str,
    draft_answer: str,
    tom: str,
    escalation: str
) -> str:
    tone_instruction = TONE_INSTRUCTIONS.get(tom, TONE_INSTRUCTIONS["neutro"])

    escalation_context = {
        AUTO_SEND: (
            "Esta resposta será enviada automaticamente ao usuário e resolve o caso."
        ),
        HUMAN_REVIEW: (
            "Esta resposta entrega valor parcial ao usuário. "
            "Um especialista da Tako já está no fluxo e irá revisar o caso. "
            "NÃO diga para o usuário buscar fontes externas."
        ),
        HUMAN_ONLY: (
            "Esta resposta NÃO resolve o caso. "
            "Ela deve apenas informar que o caso será tratado por um especialista da Tako. "
            "NÃO forneça recomendações externas, links, leis ou orientações jurídicas."
        ),
    }[escalation]

    return f"""
    Você é um assistente de comunicação da Tako, uma plataforma brasileira de automação de folha de pagamento e conformidade trabalhista.

    CONTEXTO IMPORTANTE:
    - Esta conversa acontece no WhatsApp.
    - O usuário já recebeu uma mensagem intermediária informando que a Tako está analisando o caso.
    - Esta mensagem é a resposta final do chatbot neste momento.
    - NÃO use linguagem de e-mail.
    - NÃO use saudações formais (ex: "Prezado", "Atenciosamente").
    - NÃO assine mensagens.
    - Seja direto, humano e adequado a WhatsApp.

    Mensagem original do usuário:
    \"\"\"
    {user_message}
    \"\"\"

    Resposta técnica gerada (base interna):
    \"\"\"
    {draft_answer}
    \"\"\"

    Instruções de tom:
    {tone_instruction}

    Contexto de escalonamento:
    {escalation_context}

    REGRAS OBRIGATÓRIAS:
    - NÃO recomende que o usuário consulte a CLT, advogados ou fontes externas
    - NÃO forneça aconselhamento jurídico
    - NÃO adicione informações novas
    - NÃO contradiga a resposta técnica
    - NÃO use linguagem excessivamente formal
    - NÃO use emojis

    Tarefa:
    - Reescreva a resposta para o usuário final
    - Mantenha precisão técnica
    - Linguagem natural de WhatsApp
    - Retorne APENAS o texto final da mensagem
    """


    # -------- Tone instructions --------


TONE_INSTRUCTIONS = {
    "neutro": (
        "Use um tom objetivo, claro e profissional. "
        "Não adicione empatia excessiva nem linguagem emocional."
        "A linguagem deve parecer uma mensagem de WhatsApp, não um e-mail."
    ),
    "confuso": (
        "Use um tom didático e paciente. "
        "Explique os pontos com clareza e evite termos técnicos desnecessários."
    ),
    "frustrado": (
        "Reconheça a frustração do usuário de forma respeitosa. "
        "Evite discordar ou minimizar o problema."
    ),
    "irritado": (
        "Não confronte o usuário. "
        "Valide a insatisfação, evite linguagem defensiva e ajude a reduzir o conflito."
    ),
    "urgente": (
        "Reconheça a urgência. "
        "Seja direto, organizado e transmita senso de encaminhamento."
    ),
}

# -------- Final response generator --------

def compose_final_message(
    user_message: str,
    draft_answer: str,
    escalation: str,
    tom: str
) -> str:

    if tom == "neutro" and escalation == AUTO_SEND:
        return draft_answer

    rewrite_prompt = build_rewrite_prompt(
        user_message=user_message,
        draft_answer=draft_answer,
        tom=tom,
        escalation=escalation
    )

    flow_response = invoke_flow(rewrite_prompt)
    rewritten_text = extract_flow_output(flow_response)

    rewritten_text = rewritten_text.strip()
    if not rewritten_text:
        return draft_answer

    return rewritten_text

# -------- Enviar mensagem para Z-API --------

def send_text_to_zapi(phone: str, message: str):
    print("➡️ Z-API SEND TEXT")
    ZAPI_INSTANCE_ID = os.environ["ZAPI_INSTANCE_ID"]
    ZAPI_TOKEN = os.environ["ZAPI_TOKEN"]
    ZAPI_CLIENT_TOKEN = os.environ["ZAPI_CLIENT_TOKEN"]

    ZAPI_HOST = "api.z-api.io"
    ZAPI_PATH = f"/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"

    payload = {
        "phone": phone,
        "message": message
    }

    headers = {
        "client-token": ZAPI_CLIENT_TOKEN,
        "Content-Type": "application/json"
    }

    conn = http.client.HTTPSConnection(ZAPI_HOST)

    try:
        conn.request(
            "POST",
            ZAPI_PATH,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers
        )
        response = conn.getresponse()
        response.read()

    except Exception as e:
        print(f"[ZAPI] Erro ao enviar mensagem: {e}")

    finally:
        conn.close()


# -------- Mutex no DynamoDB --------

dynamodb = boto3.resource("dynamodb")
MUTEX_TABLE_NAME = "tako_user_mutex"
mutex_table = dynamodb.Table(MUTEX_TABLE_NAME)

def acquire_user_mutex(user_id: str, ttl_seconds: int = 50) -> bool:
    expires_at = int(time.time()) + ttl_seconds

    try:
        mutex_table.put_item(
            Item={
                "user_id": user_id,
                "expires_at": expires_at
            },
            ConditionExpression="attribute_not_exists(user_id)"
        )
        return True

    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise

def release_user_mutex(user_id: str):
    try:
        mutex_table.delete_item(
            Key={"user_id": user_id}
        )
        print(f"[MUTEX] Lock liberado para {user_id}")

    except Exception as e:
        # Não pode quebrar a Lambda
        print(f"[MUTEX] Erro ao liberar lock para {user_id}: {e}")


# -------- Lambda handler --------

def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))

    user_id = body.get("phone")

    user_message = "Ocorreu um erro ao processar o input do usuario"

    if body.get("text") and body["text"].get("message"):
        user_message = body["text"]["message"]
    else:
        # Mensagem não textual (imagem, áudio, etc.)
        send_text_to_zapi(
            phone=user_id,
            message=(
                "No momento, consigo entender apenas mensagens de texto 🙂\n"
                "Pode me escrever sua dúvida aqui?"
            )
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"status": "non_text_message_ignored"})
        }

    if not user_id or not user_message:
        return {
            "statusCode": 200,
            "body": json.dumps({"status": "ignored"})
        }

    acquired = acquire_user_mutex(user_id)

    if not acquired:
        print(f"[MUTEX] Mensagem ignorada para user_id={user_id} (lock ativo)")
        return {
            "statusCode": 200,
            "body": json.dumps({"status": "locked"})
        }
    
    # Load base prompt
    base_prompt = load_prompt("message_analysis")

    # Compose final prompt
    full_prompt = f"""{base_prompt}

    User input:
    \"\"\"
    {user_message}
    \"\"\"

    """

    # Invoke Bedrock Flow
    flow_response = invoke_flow(full_prompt)

    # Extract model output (string JSON)
    output_text = extract_flow_output(flow_response)

    # Parse into structured fields
    analysis = parse_message_analysis(output_text)

    # Early response for the user
    if (analysis.get("mensagem_intermediaria") and analysis.get("intencao") != "fallback"):
        print("Calling send text function (mensagem intermediária)")
        send_text_to_zapi(
            phone=user_id,
            message=analysis["mensagem_intermediaria"]
        )

    context = {
        "user_message": user_message,
        "intencao": analysis["intencao"],
        "tom": analysis["tom"],
        "riscos": analysis["riscos"],
        "mensagem_intermediaria": analysis["mensagem_intermediaria"]
    }

    resolution = route_by_intent(
        intent=analysis["intencao"],
        context=context
    )

    escalation_decision = decide_escalation(
        tom=analysis["tom"],
        riscos=analysis["riscos"],
        confidence_score=resolution.get("confidence_score", 0)
    )

    final_message = compose_final_message(
        user_message= user_message,
        draft_answer=resolution["draft_answer"],
        escalation=escalation_decision,
        tom=analysis["tom"]
    )

    if final_message:
        print("Calling send text function 2")
        send_text_to_zapi(
            phone=user_id,
            message=final_message
        )

    release_user_mutex(user_id)

    # ADD LOG TO METADATA

    print("[FINAL RESPONSE]", json.dumps(
        {
            "user_id": user_id,
            "mensagem_intermediaria": analysis["mensagem_intermediaria"],
            "mensagem_final": final_message,
            "tom": analysis["tom"],
            "riscos": analysis["riscos"],
            "escalation": escalation_decision,
            "agent": resolution.get("agent"),
            "confidence_score": resolution.get("confidence_score")
        },
        ensure_ascii=False
    ))

    # Final response (ready for Agent 2)
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "mensagem_intermediaria": analysis["mensagem_intermediaria"],
                "mensagem_final": final_message,          # 👈 NOVO
                "tom": analysis["tom"],
                "riscos": analysis["riscos"],
                "escalation": escalation_decision,

                # Opcional (recomendo manter em staging/debug)
                "debug": {
                    "agent": resolution.get("agent"),
                    "draft_answer": resolution.get("draft_answer"),
                    "confidence_score": resolution.get("confidence_score")
                }
            },
            ensure_ascii=False
        ),
        "headers": {
            "Content-Type": "application/json"
        }
    }
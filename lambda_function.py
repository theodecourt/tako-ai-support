import json
import boto3
from pathlib import Path

# Bedrock client
bedrock = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-2"
)

FLOW_ID = "GDE6GAZUSZ"
FLOW_ALIAS_ID = "ZJ1SS18PPX"


# -------- Bedrock Flow invocation --------

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
    """
    Parses the JSON returned by the model and extracts
    intent, tone, risks and intermediate message.
    """
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

# -------- Intent-specific resolution agents --------

def handle_pagamento_atrasado(context: dict) -> dict:
    return {
        "status": "handled",
        "agent": "pagamento_atrasado_agent",
        "message": "Estamos analisando possíveis atrasos de pagamento com base nas informações disponíveis."
    }


def handle_erro_folha(context: dict) -> dict:
    return {
        "status": "handled",
        "agent": "erro_folha_agent",
        "message": "Vamos verificar possíveis inconsistências nos cálculos da folha de pagamento."
    }


def handle_demissao_rescisao(context: dict) -> dict:
    return {
        "status": "handled",
        "agent": "demissao_rescisao_agent",
        "message": "Estamos analisando as informações relacionadas ao processo de rescisão."
    }


def handle_conformidade_legal(context: dict) -> dict:
    return {
        "status": "handled",
        "agent": "conformidade_legal_agent",
        "message": "Vamos verificar a legislação aplicável para te responder corretamente."
    }


def handle_duvida_geral(context: dict) -> dict:
    return {
        "status": "handled",
        "agent": "duvida_geral_agent",
        "message": "Vamos analisar sua dúvida e te responder em seguida."
    }

def handle_fallback(context: dict) -> dict:
    return {
        "status": "fallback",
        "agent": "human_escalation",
        "message": "Vamos encaminhar sua mensagem para um especialista."
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



# -------- Lambda handler --------

def lambda_handler(event, context):
    user_message = event.get("message", "Meu salário não caiu ainda")

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

    # EARLY RESPONSE FOR THE USER

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

    # Final response (ready for Agent 2)
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "mensagem_intermediaria": analysis["mensagem_intermediaria"],
                "resolution": resolution
            },
            ensure_ascii=False
        ),
        "headers": {
            "Content-Type": "application/json"
        }
    }


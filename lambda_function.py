import json
import boto3
from pathlib import Path

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

    escalation_decision = decide_escalation(
        tom=analysis["tom"],
        riscos=analysis["riscos"],
        confidence_score=resolution.get("confidence_score", 0)
    )

    # Final response (ready for Agent 2)
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "mensagem_intermediaria": analysis["mensagem_intermediaria"],
                "riscos": analysis["riscos"],         
                "resolution": resolution,
                "escalation": escalation_decision
            },
            ensure_ascii=False
        ),
        "headers": {
            "Content-Type": "application/json"
        }
    }



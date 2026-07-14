"""
Ollama LLM Service — wraps the Resulticks internal Ollama endpoint.

Endpoint : http://10.102.1.2:7557/api/resgenapis/v2
Model    : qwen2.5:14b
Protocol : POST  {"modelname": ..., "prompt": ..., "frameworktype": "ollama"}

Handles both streaming (line-delimited JSON) and non-streaming responses.
"""

import json
import re
import logging
from typing import List, Optional

import httpx

from app.models.schemas import InsightObject, InsightType, InsightCategory, ScopeLevel

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://10.102.1.2:7557/api/resgenapis/v2"
OLLAMA_MODEL = "qwen2.5:14b"

_SYSTEM = """You are a Campaign Performance Intelligence Engine for a multi-tenant marketing platform.

Analyze the provided ML-scored campaign context and generate precise, quantitative, actionable insights.

Output ONLY a valid JSON array — no preamble, no markdown fences, no explanation.

Each object in the array must follow this exact schema:
{
  "insight_type"    : one of [Performance, Root Cause, Audience, Channel, Journey, Timing, Opportunity, Anomaly, Forecast, Prescriptive],
  "category"        : one of [Opportunity, Risk, Recommendation, Benchmark, Prediction],
  "scope"           : one of [BU, Tenant, Market, Industry],
  "title"           : "Short 5-10 word headline summarising the insight",
  "observation"     : "Factual statement with at least one specific number or percentage",
  "root_cause"      : "ML-backed explanation referencing a specific score or benchmark delta",
  "recommendation"  : "Specific, executable next action — not generic advice",
  "business_impact" : "Quantified expected uplift or risk value (revenue, %, or count)",
  "confidence"      : integer between 50 and 99
}

Quality rules:
- observation MUST contain at least one specific number or percentage from the context (e.g. "23.0% open rate", "reach_score 0.055")
- root_cause MUST reference an ML score value or benchmark delta with the exact figure
- recommendation MUST be a specific executable action with a timeframe
- business_impact MUST use exact figures (e.g. "+4.2pp conversion uplift", "₹12L revenue risk") — NEVER use multipliers like "3x", "4x", "2x better" or vague phrases like "significantly improve"
- NEVER write "X times better/worse/higher/lower" — always state the actual delta or absolute value instead
- Generate 5 to 10 insights covering as many of the 10 types as possible
- Anomaly insights only when anomaly_flags are present
- No two insights of the same type with similar observations
"""

_USER_TEMPLATE = """Analyze the following ML-scored campaign context and generate insights:

{context_block}

Prioritize in this order:
1. Anomalies flagged by ML model
2. Largest benchmark deltas (positive or negative)
3. Highest cross-sell opportunity scores
4. Channels with efficiency score gap > 0.30

Return only the JSON array."""


def _decode_ollama_response(raw: str) -> str:
    """
    Resulticks Ollama proxy returns:
      {"data": {"output": "<text>", "input_tokens": N, ...}, "status": "success"}

    Also handles legacy Ollama streaming (line-delimited {"response":..., "done":bool})
    and raw text fallback.
    """
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        # Resulticks proxy format: {"data": {"output": "..."}, "status": "success"}
        if isinstance(parsed, dict):
            if "data" in parsed and isinstance(parsed["data"], dict):
                return parsed["data"].get("output", "")
            # Standard Ollama non-streaming: {"response": "...", "done": true}
            if "response" in parsed:
                return parsed["response"]
            if "text" in parsed:
                return parsed["text"]
            if "content" in parsed:
                return parsed["content"]
            if "message" in parsed:
                msg = parsed["message"]
                return msg.get("content", "") if isinstance(msg, dict) else str(msg)
        # If the whole JSON is already an array, return it as-is
        if isinstance(parsed, list):
            return raw
    except json.JSONDecodeError:
        pass

    # Streaming: line-delimited JSON chunks {"response": "token", "done": false}
    text_parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
            if "response" in chunk:
                text_parts.append(chunk["response"])
                if chunk.get("done"):
                    break
        except json.JSONDecodeError:
            return raw  # plain text
    return "".join(text_parts) if text_parts else raw


def _extract_json_array(text: str) -> List[dict]:
    """Strip markdown, find [ … ], parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1:
        log.warning("No JSON array found in Ollama response: %s", text[:300])
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed: %s | text: %s", exc, text[start:start+300])
        return []


_MULTIPLIER_RE = re.compile(r'\b\d+(\.\d+)?[xX]\s*(better|worse|higher|lower|more|less|improvement|increase|decrease)\b', re.IGNORECASE)


def _clean_multipliers(text: str) -> str:
    """Replace 'Nx better/worse/...' with a note to use exact values. LLM should do this itself but belt-and-suspenders."""
    return _MULTIPLIER_RE.sub("[use exact delta]", text)


def _to_insight(obj: dict) -> Optional[InsightObject]:
    required = {"insight_type","category","scope","observation",
                "root_cause","recommendation","business_impact","confidence"}
    if not required.issubset(obj.keys()):
        return None
    try:
        return InsightObject(
            insight_type   = InsightType(obj["insight_type"]),
            category       = InsightCategory(obj["category"]),
            scope          = ScopeLevel(obj["scope"]),
            title          = str(obj.get("title", "")),
            observation    = _clean_multipliers(str(obj["observation"])),
            root_cause     = _clean_multipliers(str(obj["root_cause"])),
            recommendation = _clean_multipliers(str(obj["recommendation"])),
            business_impact= _clean_multipliers(str(obj["business_impact"])),
            confidence     = max(50, min(99, int(obj["confidence"]))),
        )
    except Exception as exc:
        log.debug("Skipping malformed insight: %s — %s", obj, exc)
        return None


class OllamaInsightService:

    def __init__(
        self,
        url:     str = OLLAMA_URL,
        model:   str = OLLAMA_MODEL,
        timeout: int = 120,
    ):
        self.url     = url
        self.model   = model
        self.timeout = timeout

    async def generate_insights(
        self,
        context_block: str,
        min_confidence: int = 60,
    ) -> List[InsightObject]:

        full_prompt = _SYSTEM + "\n\n" + _USER_TEMPLATE.format(context_block=context_block)

        payload = {
            "modelname":     self.model,
            "prompt":        full_prompt,
            "frameworktype": "ollama",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.url, json=payload)
                resp.raise_for_status()
                raw = resp.text
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama endpoint at {self.url}. "
                f"Ensure the Resulticks LLM server is running. Detail: {exc}"
            )
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"Ollama API returned {exc.response.status_code}: {exc.response.text[:300]}")

        decoded  = _decode_ollama_response(raw)
        raw_list = _extract_json_array(decoded)

        insights = []
        for obj in raw_list:
            insight = _to_insight(obj)
            if insight and insight.confidence >= min_confidence:
                insights.append(insight)

        log.info("Ollama returned %d raw objects → %d valid insights", len(raw_list), len(insights))
        return insights

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

Analyze the provided ML-scored campaign context and produce ONE single comprehensive consolidated insight that covers ALL findings together.

Do NOT split into multiple insights. Synthesize every signal — risks, opportunities, performance gaps, timing, journey — into a single detailed strategic analysis.

Output ONLY a valid JSON object — no preamble, no markdown fences, no explanation, no array brackets.

The object must follow this exact schema:
{
  "observation"     : "Comprehensive factual summary covering all channels and key metrics with specific numbers and percentages",
  "root_cause"      : "Unified root cause analysis referencing the dominant ML scores and benchmark deltas driving overall campaign performance",
  "recommendation"  : "Prioritized multi-step action plan addressing all findings — specific, executable, with timeframes",
  "business_impact" : "Total combined net business impact quantified in revenue, uplift %, and risk value",
  "confidence"      : integer between 50 and 99
}

Quality rules:
- observation MUST cover every channel present (email, SMS, WhatsApp, etc.) and reference at least 4 specific numbers or percentages
- root_cause MUST identify the single most impactful driver and explain how secondary factors compound it
- recommendation MUST be a numbered action plan (at least 3 steps) with specific timeframes and owners
- business_impact MUST state total uplift opportunity AND total risk, then the net expected value
- confidence reflects overall ML model certainty across all scores
- NEVER use multipliers like "3x", "4x" — always state the actual delta or absolute value
"""

_USER_TEMPLATE = """Analyze the following ML-scored campaign context and produce ONE single consolidated insight covering everything:

{context_block}

Synthesize ALL of the following into a single comprehensive insight object:
- All risk signals (bounce rate, unsubscribe rate, delivery failures, anomalies)
- All opportunity signals (high-performing channels, cross-sell score, audience fit)
- All performance gaps (underperforming channels, low ML scores, benchmark deltas)
- Timing quality, journey effectiveness, and frequency risk
- Overall conversion probability and strategic direction

Return only a single JSON object (not an array)."""


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


def _extract_json_object(text: str) -> dict:
    """Strip markdown, find { … }, parse as a single JSON object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    # If the LLM wrapped it in an array anyway, unwrap the first item
    if text.lstrip().startswith("["):
        try:
            items = json.loads(text)
            if isinstance(items, list) and items:
                return items[0] if isinstance(items[0], dict) else {}
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        log.warning("No JSON object found in Ollama response: %s", text[:300])
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        log.warning("JSON object parse failed: %s | text: %s", exc, text[start:start+300])
        return {}


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

    async def generate_consolidated_insight(
        self,
        context_block: str,
    ) -> dict:
        """Return ONE consolidated insight dict with observation/root_cause/recommendation/business_impact/confidence."""
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
                f"Cannot reach Ollama endpoint at {self.url}. Detail: {exc}"
            )
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"Ollama API returned {exc.response.status_code}: {exc.response.text[:300]}")

        decoded = _decode_ollama_response(raw)
        obj     = _extract_json_object(decoded)

        required = {"observation", "root_cause", "recommendation", "business_impact", "confidence"}
        if not required.issubset(obj.keys()):
            missing = required - obj.keys()
            raise ValueError(f"Consolidated insight missing fields: {missing}. Raw: {decoded[:400]}")

        return {
            "observation":     _clean_multipliers(str(obj["observation"])),
            "root_cause":      _clean_multipliers(str(obj["root_cause"])),
            "recommendation":  _clean_multipliers(str(obj["recommendation"])),
            "business_impact": _clean_multipliers(str(obj["business_impact"])),
            "confidence":      max(50, min(99, int(obj["confidence"]))),
        }

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

        # Hard cap: return at most 5 consolidated insights
        insights = insights[:5]

        log.info("Ollama returned %d raw objects → %d valid insights", len(raw_list), len(insights))
        return insights

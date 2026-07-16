"""
LLM Service — sends structured context to Claude and parses the insight JSON.
Enforces schema validation and confidence gating.
"""
import json
import re
from typing import List
import anthropic

from app.models.schemas import InsightObject, InsightType, InsightCategory, ScopeLevel
from app.core.config import get_settings

settings = get_settings()

SYSTEM_PROMPT_TEMPLATE = """You are a trained Campaign Performance Intelligence Engine for {tenant_name} \
operating within the {industry_name} industry, {market_name} market, {bu_name} business unit.

You have been provided a structured ML-scored campaign context. Your job is to \
analyze this context and generate precise, quantitative, actionable insights.

Hierarchy-aware rules:
- When scores fall below BU benchmarks, flag as BU-level underperformance
- When scores are above industry average but below BU average, note the gap
- Always reference the most specific benchmark level available
- Cross-sell and opportunity insights must reference the BU product catalog ({product_category})

Output ONLY a valid JSON array. No preamble. No markdown code fences. No explanation.
Each object must follow this exact schema:

{{
  "insight_type"    : one of [Performance, Root Cause, Audience, Channel, Journey, Timing, Opportunity, Anomaly, Forecast, Prescriptive],
  "category"        : one of [Opportunity, Risk, Recommendation, Benchmark, Prediction],
  "scope"           : one of [BU, Tenant, Market, Industry],
  "observation"     : "Factual statement of what happened with numbers",
  "root_cause"      : "ML-backed explanation referencing a specific score or benchmark delta",
  "recommendation"  : "Specific, executable next action (not generic advice)",
  "business_impact" : "Quantified expected uplift or risk value (revenue, %, or count)",
  "confidence"      : integer between 50 and 99
}}

Quality rules:
- Every observation MUST contain at least one number or percentage
- Root cause MUST reference an ML score value or benchmark delta
- Recommendation MUST be a specific executable action
- Business impact MUST be quantified
- Confidence reflects ML model certainty (not LLM confidence)
- No two insights of the same type should have similar observations
- Anomaly insights only when anomaly_flags list is non-empty
- Forecast insights must reference trend or day-of-campaign data
- Prescriptive insights must cite which ML score triggered the recommendation
- Suppress insights with confidence below {min_confidence}

Generate exactly {min_insights} to {max_insights} high-level consolidated insights. Do NOT produce one insight per channel or metric — GROUP related signals (e.g. all risk signals into one, all opportunity signals into one, all performance gaps into one). Each insight must synthesize multiple findings into a single strategic point.
"""

USER_PROMPT_TEMPLATE = """Analyze the following ML-scored campaign context and generate insights:

{context_block}

Prioritize in this order:
1. Anomalies flagged by the ML model
2. Largest benchmark deltas (positive or negative)
3. Highest cross-sell opportunity scores
4. Channels with efficiency score gap > 0.30

Return only the JSON array."""


def _clean_json_response(raw: str) -> str:
    """Strip any accidental markdown fences or leading/trailing text."""
    raw = raw.strip()
    # Remove ```json ... ``` wrapping if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Find the first [ ... ] block
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return raw


def _validate_insight(obj: dict) -> bool:
    required = {"insight_type", "category", "scope", "observation", "root_cause", "recommendation", "business_impact", "confidence"}
    return required.issubset(obj.keys())


class LLMService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate_insights(
        self,
        context_block: str,
        tenant_name: str,
        industry_name: str,
        market_name: str,
        bu_name: str,
        product_category: str,
    ) -> List[InsightObject]:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tenant_name=tenant_name,
            industry_name=industry_name,
            market_name=market_name,
            bu_name=bu_name,
            product_category=product_category,
            min_confidence=settings.insight_min_confidence,
            min_insights=settings.min_insights,
            max_insights=settings.max_insights,
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(context_block=context_block)

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text
        cleaned = _clean_json_response(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")

        if not isinstance(data, list):
            raise ValueError(f"LLM response is not a JSON array: {type(data)}")

        insights = []
        for obj in data:
            if not _validate_insight(obj):
                continue
            conf = int(obj.get("confidence", 0))
            if conf < settings.insight_min_confidence:
                continue
            try:
                insight = InsightObject(
                    insight_type=InsightType(obj["insight_type"]),
                    category=InsightCategory(obj["category"]),
                    scope=ScopeLevel(obj["scope"]),
                    observation=obj["observation"],
                    root_cause=obj["root_cause"],
                    recommendation=obj["recommendation"],
                    business_impact=obj["business_impact"],
                    confidence=max(50, min(99, conf)),
                )
                insights.append(insight)
            except Exception:
                continue  # skip malformed objects silently

        return insights

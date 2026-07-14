"""
Context Builder — assembles the structured ML-scored text block
that is sent to the LLM. Rich context = specific, non-generic insights.
"""
from app.models.schemas import (
    HierarchyContext, CampaignMetadata, CampaignMetrics,
    AudienceFeatures, MLScores, HistoricalContext
)


def _fmt(v, suffix="", decimals=1):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{round(v * 100, decimals)}{suffix}"
    return f"{v}{suffix}"


def _score_label(score: float) -> str:
    if score >= 0.80:
        return "[OK] High"
    if score >= 0.60:
        return "[OK]"
    if score >= 0.45:
        return "[WARNING]"
    return "[ALERT] Low"


def build_context_block(
    hierarchy: HierarchyContext,
    metadata: CampaignMetadata,
    metrics: CampaignMetrics,
    audience: AudienceFeatures,
    ml_scores: MLScores,
    historical: HistoricalContext,
) -> str:
    ch = metrics.channel_metrics
    fallback_note = ""
    if hierarchy.fallback_used:
        fallback_note = f" (fallback: {hierarchy.fallback_reason})"

    lines = [
        "=== CAMPAIGN PERFORMANCE CONTEXT ===",
        "",
        "[HIERARCHY]",
        f"Industry  : {hierarchy.industry.name}",
        f"Market    : {hierarchy.market.name}",
        f"Tenant    : {hierarchy.tenant.name}",
        f"BU        : {hierarchy.business_unit.name}",
        f"ML Scope  : {ml_scores.scope_level.value}-level model {ml_scores.model_version}{fallback_note}",
        "",
        "[CAMPAIGN METADATA]",
        f"Name           : {metadata.campaign_name}",
        f"Objective      : {metadata.campaign_objective}",
        f"Type           : {metadata.campaign_type}",
        f"Offer          : {metadata.offer_category or 'N/A'}"
        + (f" ({metadata.discount_pct}% discount)" if metadata.discount_pct else ""),
        f"Product        : {metadata.product_category}",
        f"Duration       : {metadata.campaign_duration_days} days",
        f"Execution Day  : {metadata.execution_day_of_week}",
        f"Execution Hour : {metadata.execution_hour}:00",
        f"Channels Used  : {', '.join(metadata.channels_used)}",
        f"Journey Steps  : {metadata.journey_step_count}",
        f"Comm Frequency : {metadata.communication_frequency}x",
        "",
        "[AUDIENCE SNAPSHOT]",
        f"Segment Size       : {audience.segment_size:,}",
        f"Age Mix            : 18-24: {_fmt(audience.age_18_24_pct, '%')}, "
        f"25-34: {_fmt(audience.age_25_34_pct, '%')}, "
        f"35-44: {_fmt(audience.age_35_44_pct, '%')}, "
        f"45+: {_fmt(audience.age_45_plus_pct, '%')}",
        f"Avg CLV            : {audience.avg_clv:,.0f}",
        f"Avg Churn Prob     : {audience.avg_churn_probability:.2f}",
        f"Channel Affinity   : WhatsApp: {audience.channel_affinity_whatsapp:.2f}, "
        f"Email: {audience.channel_affinity_email:.2f}, "
        f"SMS: {audience.channel_affinity_sms:.2f}",
        f"Engagement Score   : {audience.engagement_score:.1f} / 10",
        f"Propensity Score   : {audience.propensity_score:.2f}",
        "",
        "[RAW CAMPAIGN METRICS]",
        f"Reach              : {_fmt(metrics.reach_rate, '%')}  "
        + (f"(Target: {_fmt(metrics.reach_target, '%')})" if metrics.reach_target else ""),
        f"Conversion Rate    : {_fmt(metrics.overall_conversion_rate, '%')}  "
        + (f"(Target: {_fmt(metrics.conversion_target, '%')})" if metrics.conversion_target else ""),
    ]

    if ch.email_open_rate is not None:
        lines.append(f"Email Open         : {_fmt(ch.email_open_rate, '%')}")
    if ch.email_click_rate is not None:
        lines.append(f"Email Click        : {_fmt(ch.email_click_rate, '%')}")
    if ch.email_bounce_rate is not None:
        lines.append(f"Email Hard Bounce  : {_fmt(ch.email_bounce_rate, '%')}")
    if ch.email_unsubscribe_rate is not None:
        lines.append(f"Unsubscribes       : {_fmt(ch.email_unsubscribe_rate, '%')}")
    if ch.email_spam_rate is not None:
        lines.append(f"Spam Complaints    : {_fmt(ch.email_spam_rate, '%')}")
    if ch.whatsapp_open_rate is not None:
        lines.append(f"WhatsApp Open      : {_fmt(ch.whatsapp_open_rate, '%')}")
    if ch.whatsapp_click_rate is not None:
        lines.append(f"WhatsApp Click     : {_fmt(ch.whatsapp_click_rate, '%')}")
    if ch.sms_click_rate is not None:
        lines.append(f"SMS Click          : {_fmt(ch.sms_click_rate, '%')}")
    if metrics.total_revenue is not None:
        lines.append(f"Revenue            : {metrics.total_revenue:,.0f}")

    sc = ml_scores
    lines += [
        "",
        f"[ML SCORES - {sc.scope_level.value} MODEL {sc.model_version}]",
        f"reach_score                : {sc.reach_score:.2f}  {_score_label(sc.reach_score)}",
        f"engagement_quality_score   : {sc.engagement_quality_score:.2f}  {_score_label(sc.engagement_quality_score)}",
    ]
    if sc.channel_efficiency_email is not None:
        lines.append(f"channel_efficiency_email   : {sc.channel_efficiency_email:.2f}  {_score_label(sc.channel_efficiency_email)}")
    if sc.channel_efficiency_whatsapp is not None:
        lines.append(f"channel_efficiency_whatsapp: {sc.channel_efficiency_whatsapp:.2f}  {_score_label(sc.channel_efficiency_whatsapp)}")
    if sc.channel_efficiency_sms is not None:
        lines.append(f"channel_efficiency_sms     : {sc.channel_efficiency_sms:.2f}  {_score_label(sc.channel_efficiency_sms)}")
    lines += [
        f"audience_fit_score         : {sc.audience_fit_score:.2f}  {_score_label(sc.audience_fit_score)}",
        f"timing_quality_score       : {sc.timing_quality_score:.2f}  {_score_label(sc.timing_quality_score)}",
        f"journey_effectiveness      : {sc.journey_effectiveness:.2f}  {_score_label(sc.journey_effectiveness)}",
        f"frequency_risk_score       : {sc.frequency_risk_score:.2f}  {_score_label(sc.frequency_risk_score)}",
        f"churn_signal_score         : {sc.churn_signal_score:.2f}  {_score_label(sc.churn_signal_score)}",
        f"cross_sell_opportunity     : {sc.cross_sell_opportunity:.2f}  {_score_label(sc.cross_sell_opportunity)}",
        f"conversion_probability     : {sc.conversion_probability:.2f}  {_score_label(sc.conversion_probability)}",
        f"model_confidence           : {sc.model_confidence:.2f}",
    ]

    lines += ["", "[BENCHMARK DELTAS]"]
    if sc.benchmark_delta_vs_bu is not None:
        lines.append(f"vs BU Average          : Conversion {sc.benchmark_delta_vs_bu:+.1f}pp")
    if sc.benchmark_delta_vs_tenant is not None:
        lines.append(f"vs Tenant Average      : Conversion {sc.benchmark_delta_vs_tenant:+.1f}pp")
    if sc.benchmark_delta_vs_market is not None:
        lines.append(f"vs Market Average      : Conversion {sc.benchmark_delta_vs_market:+.1f}pp")
    if sc.benchmark_delta_vs_industry is not None:
        lines.append(f"vs Industry Average    : Conversion {sc.benchmark_delta_vs_industry:+.1f}pp")
    if sc.percentile_rank_bu is not None:
        lines.append(f"Percentile (BU)        : {sc.percentile_rank_bu:.0f}th")
    if sc.percentile_rank_industry is not None:
        lines.append(f"Percentile (Industry)  : {sc.percentile_rank_industry:.0f}th")

    if sc.anomaly_flags:
        lines += ["", "[ANOMALY FLAGS]"]
        for flag in sc.anomaly_flags:
            lines.append(f"- {flag.replace('_', ' ').title()}")

    lines += ["", "[HISTORICAL CONTEXT]"]
    if historical.avg_conversion_last_10:
        lines.append(f"Last 10 campaigns avg conversion : {_fmt(historical.avg_conversion_last_10, '%')}")
    if historical.avg_reach_last_10:
        lines.append(f"Last 10 campaigns avg reach      : {_fmt(historical.avg_reach_last_10, '%')}")
    if historical.best_performing_channel:
        lines.append(f"Best channel historically         : {historical.best_performing_channel}")
    if historical.best_day_of_week:
        lines.append(f"Best day historically (this BU)   : {historical.best_day_of_week}")
    if historical.same_segment_last_conversion:
        lines.append(
            f"Same segment last campaign        : {_fmt(historical.same_segment_last_conversion, '%')} conversion "
            f"(current: {(metrics.overall_conversion_rate - historical.same_segment_last_conversion) * 100:+.1f}pp)"
        )
    if historical.same_product_avg_reach:
        lines.append(f"Same product category avg reach   : {_fmt(historical.same_product_avg_reach, '%')}")

    return "\n".join(lines)

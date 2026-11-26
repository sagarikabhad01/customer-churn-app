"""Structured churn detection criteria, risk scoring, and recommendations."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskSignal:
    """A single evaluated churn risk or protective signal."""

    factor: str
    severity: str  # "critical", "warning", or "positive"
    detail: str
    recommendation: str = ""


RISK_TIERS = (
    (0.85, "Critical Risk", "error"),
    (0.60, "High Risk", "error"),
    (0.30, "Moderate Risk", "warning"),
    (0.00, "Low Risk", "success"),
)


def classify_risk_tier(probability: float) -> tuple[str, str]:
    """Return (tier_label, status_level) for a churn probability in [0, 1]."""
    for min_prob, label, level in RISK_TIERS:
        if probability >= min_prob:
            return label, level
    return "Low Risk", "success"


def evaluate_risk_signals(customer: dict[str, Any]) -> list[RiskSignal]:
    """Evaluate explicit, data-backed churn criteria against customer attributes."""
    signals: list[RiskSignal] = []

    contract = customer["contract"]
    support_calls = customer["support_calls"]
    payment_delay = customer["payment_delay"]
    total_spend = customer["total_spend"]
    age = customer["age"]
    usage = customer["usage"]
    last_interaction = customer["last_interaction"]
    tenure = customer["tenure"]

    # Contract length
    if contract == "Monthly":
        signals.append(
            RiskSignal(
                factor="Contract Length",
                severity="critical",
                detail="Monthly contract (highest-churn segment; no long-term lock-in).",
                recommendation="Offer an annual or quarterly plan upgrade with a loyalty discount.",
            )
        )
    elif contract == "Annual":
        signals.append(
            RiskSignal(
                factor="Contract Length",
                severity="positive",
                detail="Annual contract provides stronger retention stability.",
            )
        )

    # Support calls
    if support_calls > 5:
        signals.append(
            RiskSignal(
                factor="Support Calls",
                severity="critical",
                detail=f"{support_calls} support calls (exceeds critical threshold of >5).",
                recommendation="Escalate to senior support / account manager for root-cause resolution.",
            )
        )
    elif support_calls >= 4:
        signals.append(
            RiskSignal(
                factor="Support Calls",
                severity="warning",
                detail=f"{support_calls} support calls (approaching high-friction threshold of >5).",
                recommendation="Schedule a proactive check-in to resolve recurring issues.",
            )
        )
    elif support_calls <= 1:
        signals.append(
            RiskSignal(
                factor="Support Calls",
                severity="positive",
                detail=f"Low support friction ({support_calls} calls).",
            )
        )

    # Payment delay
    if payment_delay > 20:
        signals.append(
            RiskSignal(
                factor="Payment Delay",
                severity="critical",
                detail=f"{payment_delay} days delayed (exceeds critical threshold of >20 days).",
                recommendation="Offer flexible billing dates, payment reminders, or grace-period options.",
            )
        )
    elif payment_delay >= 15:
        signals.append(
            RiskSignal(
                factor="Payment Delay",
                severity="warning",
                detail=f"{payment_delay} days delayed (elevated billing risk; 15-20 days).",
                recommendation="Send an automated billing reminder and verify payment method.",
            )
        )
    elif payment_delay <= 5:
        signals.append(
            RiskSignal(
                factor="Payment Delay",
                severity="positive",
                detail=f"Healthy payment timeliness ({payment_delay} days).",
            )
        )

    # Total spend
    if total_spend < 500:
        signals.append(
            RiskSignal(
                factor="Total Spend",
                severity="critical",
                detail=f"${total_spend:,.0f} total spend (below retention threshold of $500).",
                recommendation="Provide targeted onboarding or value-demonstration offers to increase adoption.",
            )
        )
    elif total_spend < 600:
        signals.append(
            RiskSignal(
                factor="Total Spend",
                severity="warning",
                detail=f"${total_spend:,.0f} total spend (near lower spend boundary of $500).",
                recommendation="Highlight underused plan benefits or bundle incentives.",
            )
        )
    else:
        signals.append(
            RiskSignal(
                factor="Total Spend",
                severity="positive",
                detail=f"Strong spend commitment (${total_spend:,.0f}).",
            )
        )

    # Age segment
    if age > 50:
        signals.append(
            RiskSignal(
                factor="Customer Age Segment",
                severity="warning",
                detail=f"Age {age} (>50 segment shows elevated churn propensity).",
                recommendation="Ensure accessible support channels and simplified product workflows.",
            )
        )

    # Engagement & inactivity
    if usage <= 5:
        signals.append(
            RiskSignal(
                factor="Usage Frequency",
                severity="warning",
                detail=f"Low usage ({usage} sessions; <=5 indicates low product engagement).",
                recommendation="Trigger re-engagement campaign or guided feature walkthrough.",
            )
        )
    if last_interaction >= 20:
        signals.append(
            RiskSignal(
                factor="Last Interaction",
                severity="warning",
                detail=f"{last_interaction} days since last interaction (>=20 days inactivity).",
                recommendation="Send personalized outreach to re-establish contact.",
            )
        )
    if tenure >= 24 and usage >= 15:
        signals.append(
            RiskSignal(
                factor="Tenure & Engagement",
                severity="positive",
                detail=f"Established customer ({tenure} months tenure, {usage} usage frequency).",
            )
        )

    return signals


def detect_churn(
    model: Any,
    input_df: Any,
    customer: dict[str, Any],
    threshold: float = 0.50,
    enable_rule_override: bool = True,
) -> dict[str, Any]:
    """Run ML probability scoring combined with explicit rule-based criteria."""
    churn_prob = float(model.predict_proba(input_df)[0][1])
    signals = evaluate_risk_signals(customer)

    critical_signals = [s for s in signals if s.severity == "critical"]
    warning_signals = [s for s in signals if s.severity == "warning"]
    positive_signals = [s for s in signals if s.severity == "positive"]

    # Effective probability reflects model confidence; critical rule override flags
    # customers who breach hard risk boundaries even if the model sits near threshold.
    rule_triggered = enable_rule_override and len(critical_signals) >= 1
    is_churn = (churn_prob >= threshold) or rule_triggered

    tier_label, tier_level = classify_risk_tier(churn_prob)
    if is_churn and tier_label == "Low Risk":
        tier_label, tier_level = "Moderate Risk (Rule Triggered)", "warning"
    elif len(warning_signals) >= 2 and tier_label == "Low Risk":
        tier_label, tier_level = "Moderate Risk (Multiple Warnings)", "warning"

    recommendations = []
    for sig in critical_signals + warning_signals:
        if sig.recommendation and sig.recommendation not in recommendations:
            recommendations.append(sig.recommendation)

    return {
        "probability": churn_prob,
        "threshold": threshold,
        "is_churn": is_churn,
        "rule_triggered": rule_triggered,
        "risk_tier": tier_label,
        "tier_level": tier_level,
        "critical_signals": critical_signals,
        "warning_signals": warning_signals,
        "positive_signals": positive_signals,
        "recommendations": recommendations,
    }

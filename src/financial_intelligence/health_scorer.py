# src/financial_intelligence/health_scorer.py
import pandas as pd
import numpy as np
import json
from datetime import datetime


class FinancialHealthScorer:
    """
    Scores an MSME business across 5 dimensions.
    Each score is 0-100. Below 50 = at risk, above 75 = healthy.
    
    Scores:
    1. GST Compliance Score    — filing regularity, on-time payments
    2. Revenue Stability Score — consistency of monthly revenue
    3. Liquidity Score         — cash flow health
    4. Growth Score            — revenue trend over time
    5. Tax Efficiency Score    — ITC utilization, tax optimization
    """

    def __init__(self):
        self.score_weights = {
            "gst_compliance": 0.25,
            "revenue_stability": 0.20,
            "liquidity": 0.25,
            "growth": 0.15,
            "tax_efficiency": 0.15
        }

    def score_gst_compliance(self, gst_records: list) -> dict:
        """
        Scores GST compliance based on:
        - Filing on time (most important)
        - Late filing days (penalty risk)
        - Consistency of filing
        """
        if not gst_records:
            return {"score": 0, "details": "No GST data available"}

        total = len(gst_records)
        on_time = sum(1 for r in gst_records if r.get("filed_on_time", False))
        late_days = [r.get("late_filing_days", 0) for r in gst_records]
        avg_late = np.mean(late_days)

        # Base score from on-time rate
        on_time_rate = on_time / total
        base_score = on_time_rate * 100

        # Penalize for average late days
        if avg_late > 30:
            base_score -= 30
        elif avg_late > 15:
            base_score -= 15
        elif avg_late > 5:
            base_score -= 5

        score = max(0, min(100, base_score))

        # Risk level
        if score >= 75:
            risk = "LOW"
            message = "Excellent GST compliance. Filing on time consistently."
        elif score >= 50:
            risk = "MEDIUM"
            message = f"Some late filings detected. Average delay: {avg_late:.0f} days."
        else:
            risk = "HIGH"
            message = f"Poor GST compliance. {total - on_time} late filings out of {total}. Penalty risk is high."

        return {
            "score": round(score, 1),
            "risk": risk,
            "message": message,
            "on_time_rate": round(on_time_rate * 100, 1),
            "avg_late_days": round(avg_late, 1),
            "total_filings": total
        }

    def score_revenue_stability(self, gst_records: list) -> dict:
        """
        Scores revenue stability based on:
        - Coefficient of variation (lower = more stable)
        - Month-over-month drops
        - Seasonal adjustment
        """
        if len(gst_records) < 3:
            return {"score": 50, "details": "Insufficient data for stability analysis"}

        revenues = [r.get("taxable_value", 0) for r in gst_records]
        mean_rev = np.mean(revenues)
        std_rev = np.std(revenues)

        if mean_rev == 0:
            return {"score": 0, "details": "Zero revenue detected"}

        # Coefficient of variation (lower is better)
        cv = (std_rev / mean_rev) * 100

        # Score based on CV
        if cv < 10:
            score = 95
            stability = "Very Stable"
        elif cv < 20:
            score = 80
            stability = "Stable"
        elif cv < 35:
            score = 65
            stability = "Moderate"
        elif cv < 50:
            score = 45
            stability = "Unstable"
        else:
            score = 25
            stability = "Very Unstable"

        # Check for severe drops (>30% in any month)
        severe_drops = 0
        for i in range(1, len(revenues)):
            if revenues[i-1] > 0:
                drop = (revenues[i-1] - revenues[i]) / revenues[i-1]
                if drop > 0.30:
                    severe_drops += 1

        score -= severe_drops * 10
        score = max(0, min(100, score))

        return {
            "score": round(score, 1),
            "stability": stability,
            "coefficient_of_variation": round(cv, 1),
            "average_monthly_revenue": round(mean_rev, 0),
            "severe_drops": severe_drops,
            "message": f"{stability} revenue. CV: {cv:.1f}%. {severe_drops} severe drops detected."
        }

    def score_liquidity(self, gst_records: list) -> dict:
        """
        Estimates liquidity from GST data:
        - ITC vs output tax ratio
        - Net cash after tax payments
        - Buffer estimation
        """
        if not gst_records:
            return {"score": 50, "details": "No data"}

        recent = gst_records[-3:]  # Last 3 months

        total_revenue = sum(r.get("taxable_value", 0) for r in recent)
        total_gst_paid = sum(r.get("net_gst_payable", 0) for r in recent)
        total_itc = sum(r.get("itc_utilized", 0) for r in recent)

        if total_revenue == 0:
            return {"score": 0, "details": "Zero revenue"}

        # Tax burden ratio (lower is better)
        tax_burden = (total_gst_paid / total_revenue) * 100

        # ITC efficiency (higher is better — means more input credit)
        itc_ratio = (total_itc / total_revenue) * 100 if total_revenue > 0 else 0

        # Score
        if tax_burden < 5:
            score = 85
        elif tax_burden < 10:
            score = 70
        elif tax_burden < 15:
            score = 55
        elif tax_burden < 20:
            score = 40
        else:
            score = 25

        # Boost for good ITC utilization
        if itc_ratio > 50:
            score += 10
        elif itc_ratio > 30:
            score += 5

        score = max(0, min(100, score))

        monthly_avg_tax = total_gst_paid / len(recent) if recent else 0

        return {
            "score": round(score, 1),
            "tax_burden_pct": round(tax_burden, 1),
            "itc_ratio_pct": round(itc_ratio, 1),
            "avg_monthly_tax_payment": round(monthly_avg_tax, 0),
            "message": f"Tax burden: {tax_burden:.1f}% of revenue. ITC utilization: {itc_ratio:.1f}%."
        }

    def score_growth(self, gst_records: list) -> dict:
        """
        Scores revenue growth trend over last 6-12 months.
        Uses linear regression to find trend direction.
        """
        if len(gst_records) < 4:
            return {"score": 50, "details": "Insufficient data for growth analysis"}

        revenues = [r.get("taxable_value", 0) for r in gst_records]
        x = np.arange(len(revenues))

        # Linear regression
        coeffs = np.polyfit(x, revenues, 1)
        slope = coeffs[0]
        mean_rev = np.mean(revenues)

        if mean_rev == 0:
            return {"score": 0, "details": "Zero revenue"}

        # Growth rate per month as % of mean
        monthly_growth_pct = (slope / mean_rev) * 100

        # Score
        if monthly_growth_pct > 5:
            score = 95
            trend = "Strong Growth"
        elif monthly_growth_pct > 2:
            score = 80
            trend = "Moderate Growth"
        elif monthly_growth_pct > 0:
            score = 65
            trend = "Slight Growth"
        elif monthly_growth_pct > -2:
            score = 50
            trend = "Flat"
        elif monthly_growth_pct > -5:
            score = 35
            trend = "Declining"
        else:
            score = 20
            trend = "Rapidly Declining"

        # First vs last 3 months comparison
        first_3_avg = np.mean(revenues[:3])
        last_3_avg = np.mean(revenues[-3:])
        actual_growth = ((last_3_avg - first_3_avg) / first_3_avg * 100) if first_3_avg > 0 else 0

        return {
            "score": round(score, 1),
            "trend": trend,
            "monthly_growth_rate": round(monthly_growth_pct, 2),
            "period_growth_pct": round(actual_growth, 1),
            "message": f"{trend}. Monthly growth rate: {monthly_growth_pct:.1f}%. Overall period growth: {actual_growth:.1f}%."
        }

    def score_tax_efficiency(self, gst_records: list) -> dict:
        """
        Scores how efficiently the business manages its tax:
        - ITC claims vs available
        - Avoiding excess tax payment
        """
        if not gst_records:
            return {"score": 50, "details": "No data"}

        total_itc_available = sum(r.get("itc_available", 0) for r in gst_records)
        total_itc_utilized = sum(r.get("itc_utilized", 0) for r in gst_records)

        if total_itc_available == 0:
            return {"score": 50, "details": "No ITC data available"}

        # ITC utilization rate
        utilization_rate = (total_itc_utilized / total_itc_available) * 100

        if utilization_rate >= 90:
            score = 95
            efficiency = "Excellent"
        elif utilization_rate >= 75:
            score = 80
            efficiency = "Good"
        elif utilization_rate >= 60:
            score = 65
            efficiency = "Average"
        elif utilization_rate >= 40:
            score = 45
            efficiency = "Poor"
        else:
            score = 25
            efficiency = "Very Poor"

        unclaimed_itc = total_itc_available - total_itc_utilized

        return {
            "score": round(score, 1),
            "efficiency": efficiency,
            "itc_utilization_rate": round(utilization_rate, 1),
            "unclaimed_itc": round(unclaimed_itc, 0),
            "message": f"{efficiency} ITC utilization at {utilization_rate:.1f}%. Unclaimed ITC: Rs {unclaimed_itc:,.0f}."
        }

    def calculate_overall_score(self, business_id: str,
                                 gst_records: list) -> dict:
        """
        Calculates all 5 scores and combines into overall health score.
        Returns full report with recommendations.
        """
        print(f"\n{'='*50}")
        print(f"Financial Health Report: {business_id}")
        print(f"{'='*50}")

        # Calculate all scores
        scores = {
            "gst_compliance": self.score_gst_compliance(gst_records),
            "revenue_stability": self.score_revenue_stability(gst_records),
            "liquidity": self.score_liquidity(gst_records),
            "growth": self.score_growth(gst_records),
            "tax_efficiency": self.score_tax_efficiency(gst_records)
        }

        # Weighted overall score
        overall = sum(
            scores[key]["score"] * weight
            for key, weight in self.score_weights.items()
        )

        # Overall health rating
        if overall >= 80:
            rating = "EXCELLENT"
            color = "🟢"
        elif overall >= 65:
            rating = "GOOD"
            color = "🟡"
        elif overall >= 50:
            rating = "FAIR"
            color = "🟠"
        else:
            rating = "AT RISK"
            color = "🔴"

        # Print report
        print(f"\n{color} Overall Health Score: {overall:.1f}/100 — {rating}")
        print(f"\nDetailed Scores:")
        print(f"  GST Compliance    : {scores['gst_compliance']['score']:5.1f}/100 — {scores['gst_compliance'].get('risk', 'N/A')}")
        print(f"  Revenue Stability : {scores['revenue_stability']['score']:5.1f}/100 — {scores['revenue_stability'].get('stability', 'N/A')}")
        print(f"  Liquidity         : {scores['liquidity']['score']:5.1f}/100")
        print(f"  Growth            : {scores['growth']['score']:5.1f}/100 — {scores['growth'].get('trend', 'N/A')}")
        print(f"  Tax Efficiency    : {scores['tax_efficiency']['score']:5.1f}/100 — {scores['tax_efficiency'].get('efficiency', 'N/A')}")

        print(f"\nKey Insights:")
        for key, data in scores.items():
            print(f"  • {data.get('message', '')}")

        return {
            "business_id": business_id,
            "overall_score": round(overall, 1),
            "rating": rating,
            "scores": scores,
            "generated_at": datetime.now().isoformat()
        }


# ── Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    scorer = FinancialHealthScorer()

    # Load real data
    df = pd.read_csv("data/processed/gst_returns.csv")

    for business_id in ["BIZ001", "BIZ002", "BIZ003"]:
        biz_data = df[df["business_id"] == business_id].to_dict("records")
        report = scorer.calculate_overall_score(business_id, biz_data)
        print()
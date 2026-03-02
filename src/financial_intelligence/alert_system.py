# src/financial_intelligence/alert_system.py
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ProactiveAlertSystem:
    """
    Generates proactive financial alerts WITHOUT being asked.
    Runs automatically and flags issues before they become problems.

    Alert Types:
    - CRITICAL  🔴 — Needs immediate action (penalties imminent)
    - WARNING   🟠 — Needs attention this week
    - INFO      🟡 — Good to know, plan ahead
    - POSITIVE  🟢 — Good news worth highlighting
    """

    def __init__(self):
        self.alerts = []

    def _add_alert(self, alert_type: str, category: str,
                   title: str, message: str, action: str,
                   business_id: str):
        icons = {
            "CRITICAL": "🔴",
            "WARNING": "🟠",
            "INFO": "🟡",
            "POSITIVE": "🟢"
        }
        self.alerts.append({
            "type": alert_type,
            "icon": icons.get(alert_type, "⚪"),
            "category": category,
            "title": title,
            "message": message,
            "action": action,
            "business_id": business_id,
            "generated_at": datetime.now().isoformat()
        })

    # ── GST Alerts ─────────────────────────────────────────────────
    def check_gst_filing_alerts(self, gst_records: list, business_id: str):
        if not gst_records:
            return

        # Check late filing pattern
        recent = gst_records[-3:]
        late_count = sum(1 for r in recent if not r.get("filed_on_time", True))
        avg_late = np.mean([r.get("late_filing_days", 0) for r in recent])

        if late_count >= 2:
            self._add_alert(
                "CRITICAL", "GST Compliance",
                "Repeated Late GST Filings Detected",
                f"{late_count} of your last 3 GST returns were filed late "
                f"(average {avg_late:.0f} days delay). "
                f"This pattern can trigger a GST audit and attract penalties "
                f"of Rs 50/day per return under Section 47.",
                "File all pending returns immediately and set calendar reminders "
                "for 20th of every month.",
                business_id
            )
        elif late_count == 1:
            self._add_alert(
                "WARNING", "GST Compliance",
                "Late GST Filing Detected",
                f"1 of your last 3 GST returns was filed late. "
                f"Consistent late filing attracts penalties.",
                "Ensure next return is filed before the 20th deadline.",
                business_id
            )

        # Check unclaimed ITC
        total_available = sum(r.get("itc_available", 0) for r in gst_records)
        total_utilized = sum(r.get("itc_utilized", 0) for r in gst_records)
        unclaimed = total_available - total_utilized

        if unclaimed > 50000:
            self._add_alert(
                "WARNING", "Tax Optimization",
                f"Unclaimed ITC of Rs {unclaimed:,.0f} Detected",
                f"You have Rs {unclaimed:,.0f} in Input Tax Credit available "
                f"but not yet claimed. This is money you've already paid "
                f"that can be offset against your GST liability.",
                "Review your purchase invoices and claim all eligible ITC "
                "in your next GSTR-3B filing.",
                business_id
            )

        # Advance tax reminder
        current_month = datetime.now().month
        if current_month in [3, 6, 9, 12]:
            days_to_advance_tax = 15 - datetime.now().day
            if 0 < days_to_advance_tax <= 15:
                recent_revenue = np.mean([r.get("taxable_value", 0)
                                         for r in gst_records[-3:]])
                estimated_tax = recent_revenue * 0.03  # ~3% estimated advance tax
                self._add_alert(
                    "INFO", "Advance Tax",
                    f"Advance Tax Due in {days_to_advance_tax} Days",
                    f"Advance tax payment deadline is approaching. "
                    f"Based on your recent revenue of Rs {recent_revenue:,.0f}/month, "
                    f"estimated advance tax: Rs {estimated_tax:,.0f}.",
                    f"Transfer Rs {estimated_tax:,.0f} to your tax account now "
                    f"to avoid interest under Section 234B/C.",
                    business_id
                )

    # ── Revenue Alerts ─────────────────────────────────────────────
    def check_revenue_alerts(self, gst_records: list, business_id: str):
        if len(gst_records) < 3:
            return

        revenues = [r.get("taxable_value", 0) for r in gst_records]

        # Check for sudden revenue drop
        if len(revenues) >= 2:
            last_month = revenues[-1]
            prev_month = revenues[-2]

            if prev_month > 0:
                drop_pct = ((prev_month - last_month) / prev_month) * 100
                if drop_pct > 30:
                    self._add_alert(
                        "CRITICAL", "Revenue",
                        f"Revenue Dropped {drop_pct:.0f}% Last Month",
                        f"Your revenue fell from Rs {prev_month:,.0f} to "
                        f"Rs {last_month:,.0f} — a drop of {drop_pct:.0f}%. "
                        f"This may affect your loan eligibility and "
                        f"advance tax calculations.",
                        "Review top customers for payment delays. "
                        "Consider invoice discounting for immediate cash flow.",
                        business_id
                    )
                elif drop_pct > 15:
                    self._add_alert(
                        "WARNING", "Revenue",
                        f"Revenue Declined {drop_pct:.0f}% Last Month",
                        f"Revenue dropped from Rs {prev_month:,.0f} to "
                        f"Rs {last_month:,.0f}.",
                        "Follow up on outstanding invoices and review "
                        "sales pipeline for next month.",
                        business_id
                    )

        # Check revenue growth milestone
        first_3_avg = np.mean(revenues[:3])
        last_3_avg = np.mean(revenues[-3:])
        if first_3_avg > 0:
            growth = ((last_3_avg - first_3_avg) / first_3_avg) * 100
            if growth > 25:
                self._add_alert(
                    "POSITIVE", "Revenue",
                    f"Strong Revenue Growth of {growth:.0f}%",
                    f"Your average monthly revenue grew from "
                    f"Rs {first_3_avg:,.0f} to Rs {last_3_avg:,.0f} "
                    f"— a growth of {growth:.0f}%.",
                    "Consider applying for a higher credit limit or "
                    "MUDRA Tarun loan to fund this growth.",
                    business_id
                )

        # GST threshold warning
        annual_revenue_estimate = np.mean(revenues) * 12
        if 1800000 < annual_revenue_estimate < 2000000:
            self._add_alert(
                "WARNING", "GST Compliance",
                "Approaching GST Composition Scheme Limit",
                f"Your estimated annual turnover of "
                f"Rs {annual_revenue_estimate:,.0f} is approaching "
                f"the Rs 20 lakh GST registration threshold. "
                f"If you exceed this, mandatory GST registration applies.",
                "Consult your CA about whether to voluntarily register "
                "for GST now to claim ITC benefits.",
                business_id
            )

    # ── Loan Eligibility Alerts ────────────────────────────────────
    def check_loan_eligibility_alerts(self, gst_records: list,
                                       business_profile: dict,
                                       business_id: str):
        if len(gst_records) < 6:
            return

        revenues = [r.get("taxable_value", 0) for r in gst_records[-6:]]
        avg_monthly = np.mean(revenues)
        annual_estimate = avg_monthly * 12
        on_time_rate = sum(1 for r in gst_records[-6:]
                          if r.get("filed_on_time", True)) / 6

        # MUDRA loan eligibility
        if on_time_rate >= 0.8 and avg_monthly > 100000:
            if annual_estimate < 1000000:
                category = "Shishu (up to Rs 50,000)"
                amount = 50000
            elif annual_estimate < 5000000:
                category = "Kishore (Rs 50,001 - Rs 5 Lakh)"
                amount = 500000
            else:
                category = "Tarun (Rs 5 Lakh - Rs 10 Lakh)"
                amount = 1000000

            self._add_alert(
                "POSITIVE", "Loan Eligibility",
                f"You May Qualify for MUDRA {category}",
                f"Based on your GST filing history ({on_time_rate*100:.0f}% "
                f"on-time) and average monthly revenue of "
                f"Rs {avg_monthly:,.0f}, you appear eligible for a "
                f"MUDRA loan up to Rs {amount:,.0f}.",
                "Visit your nearest bank branch with last 6 months "
                "GST returns, bank statements, and Udyam certificate.",
                business_id
            )

    # ── Cash Flow Alerts ───────────────────────────────────────────
    def check_cashflow_alerts(self, gst_records: list, business_id: str):
        if len(gst_records) < 3:
            return

        recent = gst_records[-3:]
        avg_gst_payment = np.mean([r.get("net_gst_payable", 0)
                                   for r in recent])
        avg_revenue = np.mean([r.get("taxable_value", 0) for r in recent])

        # Estimate next month's GST liability
        next_month_gst = avg_gst_payment * 1.05  # 5% buffer

        if next_month_gst > 10000:
            days_to_filing = 20 - datetime.now().day
            if days_to_filing < 0:
                days_to_filing += 30

            self._add_alert(
                "INFO", "Cash Flow",
                f"GST Payment of ~Rs {next_month_gst:,.0f} Due in ~{days_to_filing} Days",
                f"Based on your last 3 months average, your next GST "
                f"payment is estimated at Rs {next_month_gst:,.0f}. "
                f"Ensure sufficient balance in your current account.",
                f"Keep Rs {next_month_gst*1.1:,.0f} reserved in your "
                f"account for GST payment by the 20th.",
                business_id
            )

    # ── Main Alert Generator ───────────────────────────────────────
    def generate_all_alerts(self, business_id: str,
                             gst_records: list,
                             business_profile: dict) -> list:
        self.alerts = []  # Reset

        self.check_gst_filing_alerts(gst_records, business_id)
        self.check_revenue_alerts(gst_records, business_id)
        self.check_loan_eligibility_alerts(gst_records, business_profile, business_id)
        self.check_cashflow_alerts(gst_records, business_id)

        # Sort: CRITICAL first, then WARNING, INFO, POSITIVE
        priority = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "POSITIVE": 3}
        self.alerts.sort(key=lambda x: priority.get(x["type"], 4))

        return self.alerts

    def print_alerts(self, business_name: str = ""):
        print(f"\n{'='*55}")
        print(f"Proactive Alerts — {business_name}")
        print(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}")
        print(f"{'='*55}")

        if not self.alerts:
            print("✅ No alerts — business looks healthy!")
            return

        for alert in self.alerts:
            print(f"\n{alert['icon']} [{alert['type']}] {alert['title']}")
            print(f"   Category : {alert['category']}")
            print(f"   Detail   : {alert['message']}")
            print(f"   Action   : {alert['action']}")

        counts = {}
        for a in self.alerts:
            counts[a["type"]] = counts.get(a["type"], 0) + 1

        print(f"\n{'='*55}")
        print(f"Summary: ", end="")
        for t, c in counts.items():
            icons = {"CRITICAL":"🔴","WARNING":"🟠","INFO":"🟡","POSITIVE":"🟢"}
            print(f"{icons.get(t,'')} {c} {t}  ", end="")
        print(f"\n{'='*55}")


# ── Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    alert_system = ProactiveAlertSystem()

    df = pd.read_csv("data/processed/gst_returns.csv")

    with open("data/processed/business_profiles.json") as f:
        profiles = json.load(f)

    for profile in profiles:
        bid = profile["business_id"]
        gst_data = df[df["business_id"] == bid].to_dict("records")

        alerts = alert_system.generate_all_alerts(bid, gst_data, profile)
        alert_system.print_alerts(profile["name"])
# src/data_pipeline/generate_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ── Business Profiles ─────────────────────────────────────────────
BUSINESS_PROFILES = [
    {
        "id": "BIZ001",
        "name": "Sharma Textile Traders",
        "industry": "Textile",
        "gstin": "07AABCS1429B1ZP",
        "state": "Delhi",
        "registration_year": 2018,
        "monthly_revenue_base": 450000
    },
    {
        "id": "BIZ002",
        "name": "Patel Electronics",
        "industry": "Electronics Retail",
        "gstin": "24AABCP1234C1ZQ",
        "state": "Gujarat",
        "registration_year": 2019,
        "monthly_revenue_base": 280000
    },
    {
        "id": "BIZ003",
        "name": "Kumar Foods Pvt Ltd",
        "industry": "Food Processing",
        "gstin": "27AABCK5678D1ZR",
        "state": "Maharashtra",
        "registration_year": 2017,
        "monthly_revenue_base": 620000
    }
]

# ── GST Returns Generator ──────────────────────────────────────────
def generate_gst_returns(business: dict, months: int = 24) -> pd.DataFrame:
    """
    Generates realistic GSTR-3B style data for a business
    GSTR-3B = Monthly summary return filed by every GST taxpayer
    """
    records = []
    base_date = datetime(2024, 1, 1)
    base_revenue = business["monthly_revenue_base"]

    for i in range(months):
        month_date = base_date - timedelta(days=30 * (months - i))

        # Add realistic seasonal variation
        seasonal_factor = 1.0
        month_num = month_date.month

        # Festival season boost (Oct-Nov)
        if month_num in [10, 11]:
            seasonal_factor = 1.35
        # Slow season (Jan-Feb)
        elif month_num in [1, 2]:
            seasonal_factor = 0.82
        # Year-end rush (March)
        elif month_num == 3:
            seasonal_factor = 1.20

        # Add random noise (+/- 15%)
        noise = np.random.uniform(0.85, 1.15)
        monthly_revenue = base_revenue * seasonal_factor * noise

        # GST calculation (18% standard rate for most B2B)
        gst_rate = 0.18
        taxable_value = monthly_revenue
        cgst = taxable_value * (gst_rate / 2)   # Central GST
        sgst = taxable_value * (gst_rate / 2)   # State GST
        igst = 0  # Interstate GST (0 for local sales)

        # Input Tax Credit (purchases)
        itc_available = taxable_value * np.random.uniform(0.55, 0.70)
        itc_utilized = itc_available * np.random.uniform(0.85, 1.0)

        # Net GST payable after ITC
        net_gst_payable = max(0, (cgst + sgst) - itc_utilized)

        # Filing date (due date is 20th of next month)
        due_date = (month_date + timedelta(days=30)).replace(day=20)
        # Sometimes filed late (realistic!)
        late_days = random.choices([0, 0, 0, 5, 15, 30], weights=[60, 15, 10, 8, 5, 2])[0]
        filing_date = due_date + timedelta(days=late_days)

        records.append({
            "business_id": business["id"],
            "business_name": business["name"],
            "industry": business["industry"],
            "gstin": business["gstin"],
            "state": business["state"],
            "return_period": month_date.strftime("%m-%Y"),
            "taxable_value": round(taxable_value, 2),
            "cgst_collected": round(cgst, 2),
            "sgst_collected": round(sgst, 2),
            "igst_collected": round(igst, 2),
            "total_tax_collected": round(cgst + sgst + igst, 2),
            "itc_available": round(itc_available, 2),
            "itc_utilized": round(itc_utilized, 2),
            "net_gst_payable": round(net_gst_payable, 2),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "filing_date": filing_date.strftime("%Y-%m-%d"),
            "filed_on_time": late_days == 0,
            "late_filing_days": late_days
        })

    return pd.DataFrame(records)

# ── Bank Statement Generator ───────────────────────────────────────
def generate_bank_statement(business: dict, months: int = 12) -> pd.DataFrame:
    """
    Generates realistic current account bank statement
    """
    transactions = []
    base_date = datetime(2024, 1, 1)
    balance = random.uniform(200000, 500000)  # Opening balance

    EXPENSE_CATEGORIES = [
        ("Supplier Payment", 0.35),
        ("Rent", 0.08),
        ("Salary", 0.20),
        ("Utilities", 0.03),
        ("GST Payment", 0.06),
        ("Loan EMI", 0.10),
        ("Miscellaneous", 0.05),
        ("Marketing", 0.04),
        ("Transport", 0.05),
        ("Other", 0.04)
    ]

    for month in range(months):
        month_date = base_date + timedelta(days=30 * month)
        monthly_revenue = business["monthly_revenue_base"] * np.random.uniform(0.85, 1.15)

        # 3-8 income transactions per month
        num_income = random.randint(3, 8)
        for _ in range(num_income):
            amount = monthly_revenue / num_income * np.random.uniform(0.8, 1.2)
            balance += amount
            tx_date = month_date + timedelta(days=random.randint(1, 28))
            transactions.append({
                "business_id": business["id"],
                "date": tx_date.strftime("%Y-%m-%d"),
                "description": f"Payment received - {random.choice(['Customer', 'Client', 'Order'])} #{random.randint(1000, 9999)}",
                "type": "CREDIT",
                "amount": round(amount, 2),
                "balance": round(balance, 2),
                "category": "Sales Revenue"
            })

        # Expense transactions
        for category, weight in EXPENSE_CATEGORIES:
            expense = monthly_revenue * weight * np.random.uniform(0.85, 1.15)
            balance -= expense
            tx_date = month_date + timedelta(days=random.randint(1, 28))
            transactions.append({
                "business_id": business["id"],
                "date": tx_date.strftime("%Y-%m-%d"),
                "description": f"{category} - {month_date.strftime('%b %Y')}",
                "type": "DEBIT",
                "amount": round(expense, 2),
                "balance": round(max(balance, 0), 2),
                "category": category
            })

    df = pd.DataFrame(transactions)
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ── Business Profile Generator ─────────────────────────────────────
def generate_business_profile(business: dict) -> dict:
    """
    Creates a unified business profile JSON
    This is what the RAG agent uses as context
    """
    return {
        "business_id": business["id"],
        "name": business["name"],
        "industry": business["industry"],
        "gstin": business["gstin"],
        "state": business["state"],
        "registration_year": business["registration_year"],
        "monthly_revenue_base": business["monthly_revenue_base"],
        "annual_revenue_estimate": business["monthly_revenue_base"] * 12,
        "business_size": "Micro" if business["monthly_revenue_base"] < 300000 else "Small",
        "gst_registered": True,
        "created_at": datetime.now().isoformat()
    }

# ── Main Runner ────────────────────────────────────────────────────
def generate_all_data():
    print("="*50)
    print("Generating Synthetic MSME Financial Data")
    print("="*50)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    all_gst = []
    all_bank = []
    all_profiles = []

    for business in BUSINESS_PROFILES:
        print(f"\n📊 Generating data for: {business['name']}")

        # GST Returns
        gst_df = generate_gst_returns(business, months=24)
        all_gst.append(gst_df)
        print(f"   ✅ GST returns: {len(gst_df)} months generated")

        # Bank Statements
        bank_df = generate_bank_statement(business, months=12)
        all_bank.append(bank_df)
        print(f"   ✅ Bank transactions: {len(bank_df)} transactions generated")

        # Business Profile
        profile = generate_business_profile(business)
        all_profiles.append(profile)
        print(f"   ✅ Business profile created")

    # Save everything
    gst_combined = pd.concat(all_gst, ignore_index=True)
    bank_combined = pd.concat(all_bank, ignore_index=True)

    gst_combined.to_csv("data/processed/gst_returns.csv", index=False)
    bank_combined.to_csv("data/processed/bank_statements.csv", index=False)

    with open("data/processed/business_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=2)

    print("\n" + "="*50)
    print("✅ All data generated successfully!")
    print(f"   GST records: {len(gst_combined)} rows")
    print(f"   Bank records: {len(bank_combined)} rows")
    print(f"   Business profiles: {len(all_profiles)}")
    print("\nFiles saved to data/processed/")
    print("="*50)

if __name__ == "__main__":
    generate_all_data()
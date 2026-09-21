from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime, timedelta

router = APIRouter()

# Paths to the needed data/models – adjust if you move them
FIN_DATA = Path("../data/finanzas_personales.csv")  # you should provide this CSV
SALARY_MODEL = Path("../api-prediccion-salarios/models/salary_model.joblib")

# Load the salary ML model from Proyecto 09 (fallback if missing)
if SALARY_MODEL.exists():
    salary_model = joblib.load(SALARY_MODEL)
else:
    salary_model = None

class CashflowInput(BaseModel):
    # Income side (optional – you can leave empty and rely on salary model)
    predicted_monthly_income: float | None = None
    # Expenses side – list of upcoming expenses (date, amount)
    upcoming_expenses: list[dict] | None = None
    # Simulation horizon (days)
    horizon_days: int = 30

class CashflowOutput(BaseModel):
    cash_balance_history: list[dict]
    recommended_action: str
    probability_negative: float

def load_financial_history() -> pd.DataFrame:
    """Load historical cash‑flow from a CSV (date, amount)."""
    if not FIN_DATA.exists():
        raise FileNotFoundError("Financial history CSV not found.")
    df = pd.read_csv(FIN_DATA, parse_dates=["date"])
    return df.sort_values("date")

def predict_income_from_salary() -> float:
    """Simple wrapper that asks the salary prediction model for a monthly income.
    Here we use an average salary; you could pass a more detailed profile.
    """
    if salary_model is None:
        raise RuntimeError("Salary model not available.")
    # Dummy input – you could improve by passing real user profile
    dummy = pd.DataFrame([{
        "job_title": "Data Scientist",
        "experience_level": "SE",
        "employment_type": "FT",
        "remote_ratio": 100,
        "company_size": "L",
    }])
    pred = salary_model.predict(dummy)[0]
    # Assume monthly salary = annual / 12
    return float(pred) / 12

@router.post("/predict-cashflow", response_model=CashflowOutput)
async def predict_cashflow(payload: CashflowInput):
    # 1️⃣ Load historic cash‑flow
    history = load_financial_history()
    # 2️⃣ Determine base income
    if payload.predicted_monthly_income is not None:
        base_income = payload.predicted_monthly_income
    else:
        # Use salary model as fallback
        base_income = predict_income_from_salary()
    # 3️⃣ Build a daily series for the horizon
    start_date = datetime.today().date()
    dates = [start_date + timedelta(days=i) for i in range(payload.horizon_days)]
    df = pd.DataFrame({"date": dates})
    df["income"] = base_income / payload.horizon_days  # spread monthly income evenly
    df["expense"] = 0.0
    # 4️⃣ Add upcoming expenses if any
    if payload.upcoming_expenses:
        for exp in payload.upcoming_expenses:
            exp_date = pd.to_datetime(exp["date"]).date()
            mask = df["date"] == exp_date
            df.loc[mask, "expense"] += exp["amount"]
    # 5️⃣ Compute cumulative cash balance (starting from last known balance)
    last_balance = history["balance"].iloc[-1] if "balance" in history.columns else 0.0
    df["net"] = df["income"] - df["expense"]
    df["cumulative"] = last_balance + df["net"].cumsum()
    # 6️⃣ Evaluate risk of negative balance
    prob_negative = (df["cumulative"] < 0).mean()
    # 7️⃣ Simple recommendation logic
    if prob_negative > 0.5:
        action = "Consider acelerar cobros o reducir gastos próximos."
    elif prob_negative > 0.2:
        action = "Mantén vigilancia – podrías quedar sin caja en pocos días."
    else:
        action = "Flujo de caja saludable – puedes planear inversiones."
    # 8️⃣ Build response
    history_out = df[['date','income','expense','cumulative']].to_dict(orient='records')
    return CashflowOutput(
        cash_balance_history=history_out,
        recommended_action=action,
        probability_negative=round(prob_negative, 3)
    )

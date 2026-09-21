import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# ---- Configuración ----
st.set_page_config(page_title="CashFlow Prophet", page_icon="💰", layout="centered")

st.title("💰 CashFlow Prophet")
st.caption("Pronóstico y optimización de flujo de caja para freelancers y pymes")

# ---- Cargar datos históricos de finanzas ----
FIN_DATA = Path("data/finanzas_personales.csv")
if FIN_DATA.exists():
    df_hist = pd.read_csv(FIN_DATA, parse_dates=["date"])
    df_hist = df_hist.sort_values("date")
    # Asumimos columnas: date, amount (positivo ingreso, negativo gasto)
    # Calcular balance acumulado
    df_hist["balance"] = df_hist["amount"].cumsum()
else:
    st.warning("No se encontró el archivo de datos `data/finanzas_personales.csv`. "
               "Por favor, exporte su historial de finanzas desde su proyecto de finanzas personales.")
    df_hist = pd.DataFrame(columns=["date", "amount", "balance"])

# ---- Cargar modelo salarial (de Proyecto 09) ----
SALARY_MODEL_PATH = Path("../api-prediccion-salarios/models/salary_model.joblib")
if SALARY_MODEL_PATH.exists():
    salary_model = joblib.load(SALARY_MODEL_PATH)
else:
    salary_model = None
    st.info("Modelo salarial no encontrado. Se usará un ingreso estimado manual.")

# ---- Sidebar para inputs ----
st.sidebar.header("Parámetros de simulación")
horizon_days = st.sidebar.slider("Horizonte de pronóstico (días)", 7, 90, 30)

# Ingreso mensual estimado
if salary_model is not None:
    # Usar un perfil por defecto (puedes hacerlo configurable)
    dummy_input = pd.DataFrame([{
        "job_title": "Data Scientist",
        "experience_level": "SE",
        "employment_type": "FT",
        "remote_ratio": 100,
        "company_size": "L",
    }])
    pred_annual = salary_model.predict(dummy_input)[0]
    estimated_monthly_income = pred_annual / 12
    st.sidebar.success(f"Ingreso mensual estimado (modelo): ${estimated_monthly_income:,.2f}")
    base_income = estimated_monthly_income
else:
    base_income = st.sidebar.number_input(
        "Ingreso mensual estimado ($)", min_value=0.0, value=3000.0, step=100.0
    )

# Gastos futuros conocidos (opcional)
st.sidebar.subheader("Gastos futuros conocidos")
num_expenses = st.sidebar.number_input("Número de gastos futuros", min_value=0, max_value=10, value=0, step=1)
upcoming_expenses = []
for i in range(int(num_expenses)):
    exp_date = st.sidebar.date_input(f"Fecha del gasto {i+1}", value=datetime.today() + timedelta(days=7))
    exp_amount = st.sidebar.number_input(f"Monto del gasto {i+1} ($)", min_value=0.0, value=100.0, step=10.0, key=f"exp{i}")
    upcoming_expenses.append({"date": exp_date.strftime("%Y-%m-%d"), "amount": exp_amount})

# ---- Cálculo del flujo de caja ----
st.subheader("📈 Pronóstico de flujo de caja")

# Serie diaria
start_date = datetime.today().date()
dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
df_forecast = pd.DataFrame({"date": dates})
df_forecast["income"] = base_income / horizon_days  # distribuir ingreso mensual uniformemente
df_forecast["expense"] = 0.0

# Añadir gastos futuros
for exp in upcoming_expenses:
    exp_date = pd.to_datetime(exp["date"]).date()
    mask = df_forecast["date"] == exp_date
    df_forecast.loc[mask, "expense"] += exp["amount"]

# Balance inicial (último balance histórico o cero)
initial_balance = df_hist["balance"].iloc[-1] if not df_hist.empty else 0.0
df_forecast["net"] = df_forecast["income"] - df_forecast["expense"]
df_forecast["cumulative"] = initial_balance + df_forecast["net"].cumsum()

# Métricas
prob_negative = (df_forecast["cumulative"] < 0).mean()
min_balance = df_forecast["cumulative"].min()
final_balance = df_forecast["cumulative"].iloc[-1]

# Mostrar gráficos
st.line_chart(df_forecast.set_index("date")[["cumulative"]])
st.caption("Balance acumulado proyectado (línea)")

col1, col2, col3 = st.columns(3)
col1.metric("Balance final proyectado", f"${final_balance:,.2f}")
col2.metric("Balance mínimo", f"${min_balance:,.2f}")
col3.metric("Prob. de negativo", f"{prob_negative:.0%}")

# Recomendación
if prob_negative > 0.5:
    rec = "🚨 Riesgo alto de quedar sin caja. Considere facturar pendientes, reducir gastos o conseguir financiación a corto plazo."
elif prob_negative > 0.2:
    rec = "⚠️ Riesgo moderado. Revise sus gastos próximos y asegúrese de tener un colchón de seguridad."
else:
    rec = "✅ Flujo de caja saludable. Puede considerar inversiones o ahorro adicional."
st.info(rec)

# Mostrar tabla detallada si se desea
with st.expander("Ver tabla detallada"):
    st.dataframe(df_forecast)

# ---- Footer ----
st.markdown("---")
st.markdown("Creado con ❤️ usando Streamlit. Modelo salarial proveniente del Proyecto 09.")

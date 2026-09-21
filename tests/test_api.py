from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()["message"] == "Bienvenido a CashFlow Prophet API"

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

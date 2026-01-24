from fastapi.testclient import TestClient
from backend.main import app
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

client = TestClient(app)

def test_hospital_types():
    print("\nTesting /hospitals/types...")
    response = client.get("/hospitals/types")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert "types" in response.json()
    assert "Government" in response.json()['types']

def test_hospital_search():
    print("\nTesting /hospitals/search?query=Apollo&type=Private...")
    response = client.get("/hospitals/search?query=Apollo&type=Private")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Count: {data.get('count')}")
    if data.get('hospitals'):
        print(f"First Result: {data['hospitals'][0]['name']}")
    
    assert response.status_code == 200
    assert data['count'] > 0
    assert "Apollo" in data['hospitals'][0]['name']

def test_hospital_search_govt():
    print("\nTesting /hospitals/search?query=Aiims&type=Government...")
    response = client.get("/hospitals/search?query=Aiims&type=Government")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Count: {data.get('count')}")
    
    assert response.status_code == 200
    assert data['count'] > 0

if __name__ == "__main__":
    try:
        test_hospital_types()
        test_hospital_search()
        test_hospital_search_govt()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

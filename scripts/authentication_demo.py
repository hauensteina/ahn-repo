"""REST API authentication patterns demo."""

import requests

BASE_URL = "https://api.example.com"


# 1. API Key in header
def api_key_auth():
    response = requests.get(
        f"{BASE_URL}/data",
        headers={"X-API-Key": "your-api-key-here"},
    )
    return response.json()


# 2. Bearer token (OAuth2 / JWT)
def bearer_token_auth():
    token = "your-bearer-token"
    response = requests.get(
        f"{BASE_URL}/data",
        headers={"Authorization": f"Bearer {token}"},
    )
    return response.json()


# 3. Basic auth (username + password)
def basic_auth():
    response = requests.get(
        f"{BASE_URL}/data",
        auth=("username", "password"),
    )
    return response.json()


# 4. Session-based (login once, reuse cookies)
def session_auth():
    session = requests.Session()
    session.post(f"{BASE_URL}/login", json={"user": "me", "pass": "secret"})
    # session now carries the auth cookie
    response = session.get(f"{BASE_URL}/data")
    return response.json()


# 5. POST with JSON body + error handling (most realistic)
def post_with_auth():
    try:
        response = requests.post(
            f"{BASE_URL}/items",
            headers={"Authorization": "Bearer your-token"},
            json={"name": "widget", "quantity": 5},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    # Swap in a real URL and credentials to try it out
    print(post_with_auth())
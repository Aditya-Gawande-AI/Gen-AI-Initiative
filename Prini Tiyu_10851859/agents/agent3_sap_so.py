import os
import requests
import urllib3
from requests.auth import HTTPBasicAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _fetch_csrf_token(session: requests.Session, url: str) -> str:
    """
    Fetch CSRF token using HEAD with X-CSRF-Token: Fetch.
    SAP systems often bind CSRF token to session cookie, so we use the same session.
    This approach is aligned with standard SAP CSRF handling guidance. [3](https://sap.github.io/cloud-sdk/docs/js/features/connectivity/csrf)[2](https://stackoverflow.com/questions/76306516/csrf-error-after-create-entity-post-request-in-abap-odata-service)
    """
    headers = {
        "Accept": "application/json",
        "X-CSRF-Token": "Fetch"
    }

    # Try with and without trailing slash (some SAP systems differ) [3](https://sap.github.io/cloud-sdk/docs/js/features/connectivity/csrf)
    for candidate in (url.rstrip("/") + "/", url.rstrip("/")):
        r = session.head(candidate, headers=headers, verify=False, timeout=60)
        token = r.headers.get("X-CSRF-Token") or r.headers.get("x-csrf-token")
        if token and token.lower() != "fetch":
            return token

    # Some systems may not return token on HEAD; fallback to GET
    for candidate in (url.rstrip("/") + "/", url.rstrip("/")):
        r = session.get(candidate, headers=headers, verify=False, timeout=60)
        token = r.headers.get("X-CSRF-Token") or r.headers.get("x-csrf-token")
        if token and token.lower() != "fetch":
            return token

    raise RuntimeError(f"Could not fetch CSRF token from SAP endpoint: {url}")


def create_sales_order(serial_number: str):
    """
    Create Sales Order using SAP OData service.
    Fixes CSRF failure by fetching a fresh token bound to the session cookie. [1](https://community.sap.com/t5/technology-blog-posts-by-sap/issues-with-csrf-token-and-how-to-solve-them/ba-p/13100456)[2](https://stackoverflow.com/questions/76306516/csrf-error-after-create-entity-post-request-in-abap-odata-service)
    """
    url = os.getenv("SAP_SO_URL")  # e.g. https://172.19.151.9:44302/.../A_SalesOrder
    user = os.getenv("SAP_USER", "developer")
    pwd = os.getenv("SAP_PASS", "dev@S09")

    if not url:
        raise RuntimeError("Missing SAP_SO_URL in .env")

    session = requests.Session()
    session.auth = HTTPBasicAuth(user, pwd)
    session.verify = False  # internal certs

    # 1) Fetch CSRF token (and session cookie) using same session
    csrf_token = _fetch_csrf_token(session, url)

    # 2) POST using fetched token + same session cookies
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf_token
    }

    body = {
        "SalesOrderType": "OR",
        "SalesOrganization": "1710",
        "DistributionChannel": "10",
        "OrganizationDivision": "00",
        "SoldToParty": "101",
        "PurchaseOrderByCustomer": serial_number,
        "CustomerPaymentTerms": "0001",
        "to_Item": [
            {"Material": "ORDER_BOM", "RequestedQuantity": "2"}
        ]
    }

    resp = session.post(url, json=body, headers=headers, timeout=60)

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if resp.status_code >= 300:
        raise RuntimeError(f"Sales Order creation failed ({resp.status_code}):\n{data}")

    return data
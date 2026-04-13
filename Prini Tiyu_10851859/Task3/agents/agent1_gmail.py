import base64
from googleapiclient.discovery import build
from utils.gmail_auth import get_creds

def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data.encode("utf-8"))

def fetch_latest_image_attachment(query: str):
    """
    Returns: (filename, image_bytes, message_id)
    """
    service = build("gmail", "v1", credentials=get_creds())

    res = service.users().messages().list(userId="me", q=query, maxResults=5).execute()
    msgs = res.get("messages", [])
    if not msgs:
        raise RuntimeError("No Gmail messages found for the given query.")

    msg_id = msgs[0]["id"]
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()

    payload = msg.get("payload", {})
    parts = payload.get("parts", []) or []

    # Attachments are retrieved using users.messages.attachments.get [6](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages.attachments/get)
    for part in parts:
        filename = part.get("filename", "")
        body = part.get("body", {})
        att_id = body.get("attachmentId")

        if filename and att_id and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            att = service.users().messages().attachments().get(
                userId="me", messageId=msg_id, id=att_id
            ).execute()
            data = att.get("data")
            if not data:
                raise RuntimeError("Found attachment but 'data' was empty.")
            return filename, _b64url_decode(data), msg_id

    raise RuntimeError("No image attachment (.png/.jpg/.jpeg) found in the matched email.")

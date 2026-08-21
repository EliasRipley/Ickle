import json
import os
from urllib.request import Request, urlopen

from src.icklization import ick


def cloud_is_configured() -> bool:
    return bool(os.getenv("ILM_CLOUD_API_KEY"))


def cloud_status_text() -> str:
    if cloud_is_configured():
        return ick.cloud_configured_text()
    return ick.cloud_not_configured_text()


def assist(prompt: str, model: str | None = None) -> str:
    api_key = os.getenv("ILM_CLOUD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing ILM_CLOUD_API_KEY; cloud assist disabled.")

    base_url = os.getenv("ILM_CLOUD_BASE_URL", "https://api.openai.com/v1")
    endpoint = os.getenv("ILM_CLOUD_ENDPOINT", "/responses")
    url = base_url.rstrip("/") + endpoint
    req_model = model or os.getenv("ILM_CLOUD_MODEL", "gpt-4o-mini")

    payload = {
        "model": req_model,
        "input": prompt,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    # Try common response shapes robustly.
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    output = body.get("output", [])
    if output and isinstance(output, list):
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else []
            for c in content:
                text = c.get("text") if isinstance(c, dict) else None
                if text:
                    return text
    # Chat completions compatibility fallback.
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"])

    return json.dumps(body)

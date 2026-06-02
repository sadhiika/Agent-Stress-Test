import os
from google import genai

PROJECT = "project-0c4b55b6-7d9b-4fd0-b6a"
LOCATION = "us-central1"
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
resp = client.models.generate_content(
    model=MODEL,
    contents="Reply with exactly: hello from vertex",
)
print("MODEL:", MODEL)
print("TEXT:", resp.text)
print("USAGE:", resp.usage_metadata)
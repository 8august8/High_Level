from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="who are you?",
    # config=types.GenerateContentConfig(
    #     thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    # ),
)
print(response.text)

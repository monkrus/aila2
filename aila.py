import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import mesop as me
import mesop.labs as mel

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Ensure API key is available
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    logging.error("NVIDIA_API_KEY environment variable not found.")
    raise EnvironmentError("NVIDIA_API_KEY environment variable not found.")

# Initialize OpenAI client with NVIDIA endpoint
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/",
    title="Mesop Demo Chat",
)
def page():
    mel.chat(transform, title="Mesop Demo Chat", bot_user="Mesop Bot")

def transform(input: str, history: list[mel.ChatMessage]):
    try:
        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{"role": "user", "content": input}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )
        
        response = ""
        previous_content = ""

        for chunk in completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                # Remove any leading duplicate parts that might be carried over
                if response.endswith(previous_content):
                    response = response[:-len(previous_content)]
                response += delta_content
                previous_content = delta_content
                yield response
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        yield "An error occurred while processing your request. Please try again."

if __name__ == "__main__":
    me.run()

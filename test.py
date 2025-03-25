from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("The environment variable 'OPENAI_API_KEY' is not set.")

os.environ["OPENAI_API_KEY"] = api_key

print("api", api_key)
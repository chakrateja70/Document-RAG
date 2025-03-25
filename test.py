from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv('PINECONE_API_KEY')

if not api_key:
    raise ValueError("The environment variable 'PINECONE_API_KEY' is not set. Please set it before running the script.")

print("api_key", api_key)
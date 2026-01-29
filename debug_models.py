import os
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: API Key not found. Check your .env file.")
else:
    try:
        client = Groq(api_key=api_key)
        print("🔄 Connecting to Groq API...")
        
        # Fetch list of available models
        models = client.models.list()
        
        print("\n✅ ACCESS GRANTED. HERE ARE YOUR AVAILABLE MODELS:")
        print("="*50)
        found_vision = False
        for model in models.data:
            print(f"🔹 {model.id}")
            if "vision" in model.id:
                found_vision = True
        print("="*50)
        
        if found_vision:
            print("🎉 GOOD NEWS: You HAVE Vision models! Use one of the IDs above.")
        else:
            print("⚠️ BAD NEWS: Your account has NO Vision models enabled.")
            print("👉 Solution: Go to console.groq.com and create a NEW API Key.")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
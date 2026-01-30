import os
import logging
import sys
from dotenv import load_dotenv
from groq import Groq, AuthenticationError, APIConnectionError

# --- LOGGING CONFIGURATION ---
# Sets up a standard logging format used in production environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def validate_model_access():
    """
    Diagnoses Groq API connectivity and verifies available model endpoints.
    Returns:
        bool: True if Vision models are available, False otherwise.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logging.critical("Configuration Error: GROQ_API_KEY not found in environment variables.")
        return False

    try:
        logging.info("Initializing handshake with Groq API...")
        client = Groq(api_key=api_key)
        
        # Fetch available models
        models = client.models.list()
        
        vision_capable = False
        available_models = []

        logging.info("--- AVAILABLE MODEL ENDPOINTS ---")
        for model in models.data:
            # Log model IDs for debugging configuration
            logging.info(f"Endpoint: {model.id}")
            available_models.append(model.id)
            
            if "vision" in model.id.lower():
                vision_capable = True
        
        logging.info("---------------------------------")

        if vision_capable:
            logging.info("✅ System Check Passed: Vision-capable endpoints are active.")
            return True
        else:
            logging.warning("⚠️ Configuration Warning: No Vision models detected in this API tier.")
            logging.warning("Action Required: verify API key permissions or model availability region.")
            return False

    except AuthenticationError:
        logging.error("❌ Authentication Failed: Invalid API Key. Please rotate your credentials.")
        return False
    except APIConnectionError:
        logging.error("❌ Network Error: Unable to connect to Groq API gateway.")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected Runtime Error: {str(e)}")
        return False

if __name__ == "__main__":
    validate_model_access()

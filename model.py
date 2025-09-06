from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

try:
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Add token="your_token_here" for private repos
    # Load model
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)  # Add token="your_token_here" for private repos
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {str(e)}")
    raise
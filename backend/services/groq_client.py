from groq import Groq
from backend.config import GROQ_API_KEYS, GROQ_MODEL

class RotatingGroqClient:
    def __init__(self, api_keys: list[str]):
        self.api_keys      = api_keys
        self.current_index = 0
        self.client        = Groq(api_key=self.api_keys[0])
        self.retry_count   = 0
        self.max_retries   = len(api_keys)  # Max retries = number of keys
        print(f"[Groq] Initialized with {len(api_keys)} API key(s)")

    def _rotate(self):
        self.current_index += 1
        if self.current_index >= len(self.api_keys):
            raise Exception("❌ All Groq API keys exhausted. Please check your API key configuration and try again later.")
        self.client = Groq(api_key=self.api_keys[self.current_index])
        print(f"[Groq] Rotated to API key {self.current_index + 1}/{len(self.api_keys)}")

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 1500, attempt: int = 0) -> str:
        try:
            # Validate inputs
            if not messages or not isinstance(messages, list):
                raise ValueError("messages must be a non-empty list")
            if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2.0:
                temperature = 0.1
            if not isinstance(max_tokens, int) or max_tokens < 1:
                max_tokens = 1500
            
            response = self.client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens,
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from Groq API")
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e).lower()
            
            # Rate limit error - try next key
            if "429" in str(e) or "rate_limit" in error_str or "quota" in error_str:
                print(f"[Groq] Rate limit/quota hit on key {self.current_index + 1}")
                if attempt < self.max_retries:
                    try:
                        self._rotate()
                        return self.chat(messages, temperature, max_tokens, attempt + 1)
                    except Exception as rotate_err:
                        raise Exception(f"Rate limit - all API keys exhausted: {rotate_err}")
                else:
                    raise Exception("Rate limit - max retries exceeded")
            
            # Authentication error
            elif "401" in str(e) or "unauthorized" in error_str or ("invalid" in error_str and "api" in error_str):
                raise Exception(f"Groq API authentication failed. Please check your API keys. Details: {e}")
            
            # Other API errors
            elif "500" in str(e) or "503" in str(e):
                raise Exception(f"Groq API server error. Please try again later. Details: {e}")
            else:
                raise Exception(f"Groq Error: {e}")

# Single shared instance used by all services
groq = RotatingGroqClient(api_keys=GROQ_API_KEYS)
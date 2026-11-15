from llamea import Ollama_LLM

import ollama

import time

class OllamaWrapper(Ollama_LLM):
    def __init__(self, model="llama3.2", **kwargs):
        super().__init__(model)
        # self.temperature = temperature
        # self.num_ctx = num_ctx
        self.options = kwargs

    def query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        # first concatenate the session messages
        big_message = ""
        for msg in session_messages:
            big_message += msg["content"] + "\n"

        attempt = 0
        while True:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": big_message}],
                    options=self.options
                    # options={"temperature": self.temperature, "num_ctx": self.num_ctx}

                )
                return response["message"]["content"]

            except ollama.ResponseError as err:
                attempt += 1
                if attempt > max_retries or err.status_code not in (429, 500, 503):
                    raise
                time.sleep(default_delay * attempt)

            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)
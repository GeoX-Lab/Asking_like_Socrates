import base64
import os
import time
from io import BytesIO
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from volcenginesdkarkruntime import Ark

# Load environment variables
load_dotenv()

# Configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
ARK_API_KEY = os.getenv("ARK_API_KEY")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

MAX_RETRIES = 3


def encode_pil_image(image: Image.Image, format: str) -> str:
    """Encodes a PIL image to a base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def chat_completions(
    model: str,
    query: str,
    images: Optional[List[Tuple[Any, str]]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Calls OpenAI Chat Completion API. Supports text-only and multimodal inputs.

    Args:
        model: Model identifier (e.g., 'gpt-4o').
        query: User query text.
        images: Optional list of (PIL.Image, suffix) tuples.
        system_prompt: Optional system instruction.

    Returns:
        The model's response content.
    """
    client = OpenAI(
        api_key=OPENAI_KEY,
        base_url=OPENAI_BASE_URL,
    )

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    user_content = []

    # Process images if provided
    if images:
        for image, suffix in images:
            base64_data = encode_pil_image(image, format=suffix.upper())
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{suffix.lower()};base64,{base64_data}"
                }
            })

    user_content.append({"type": "text", "text": query})

    messages.append({
        "role": "user",
        "content": user_content
    })

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    return completion.choices[0].message.content


def chat_doubao(
    model: str,
    query: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Calls Volcengine (Doubao) API for text-only interactions.
    """
    client = Ark(
        base_url=DOUBAO_BASE_URL,
        api_key=ARK_API_KEY,
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": query})

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return completion.choices[0].message.content


class APIModel:
    """
    Wrapper class to handle multi-backend LLM calls with retry logic.
    """

    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _single_call(
        self, 
        query: str, 
        images: Optional[List[Tuple[Image.Image, str]]] = None
    ) -> str:
        """Dispatches the call to the appropriate provider based on model name."""
        if self.model_name in ["gemini-2.5-flash", "gpt-5-mini"]:
            return chat_completions(
                model=self.model_name,
                query=query,
                images=images,
                system_prompt=self.system_prompt,
            )
        elif self.model_name == "doubao-seed-1-6-thinking-250715":
            # Note: Current implementation ignores images for Doubao
            return chat_doubao(self.model_name, query, self.system_prompt)
        else:
            raise NotImplementedError(f"Unknown model: {self.model_name}")

    def get_response(
        self, 
        query: str, 
        images: Optional[List[Tuple[Image.Image, str]]] = None
    ) -> str:
        """
        Executes the API call with automatic retries.

        Args:
            query: Input text.
            images: Optional list of (PIL.Image, suffix) tuples.

        Returns:
            The model response string.
        """
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_call(query, images)
            except Exception as e:
                last_error = e
                print(f"[API RETRY] {self.model_name} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                
                if attempt < MAX_RETRIES:
                    time.sleep(0.5)
                else:
                    raise last_error


if __name__ == "__main__":
    model_name = "gemini-2.5-flash"
    # model_name = "gpt-5-mini"
    # model_name = "doubao-seed-1-6-thinking-250715"

    model = APIModel(model_name, "用小朋友的语气回答我")
    response = model.get_response("维斯塔潘获得过几届世界冠军？")
    print(response)

    print("-" * 20)

    # Ensure this path exists or mock it for testing
    image_path = "assets/Fig1.jpg"
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        model = APIModel(model_name, "用小朋友的语气回答我")
        response = model.get_response("这张图描述了什么？", images=[(img, 'jpeg')])
        print(response)
    else:
        print(f"Test image not found at {image_path}, skipping image test.")
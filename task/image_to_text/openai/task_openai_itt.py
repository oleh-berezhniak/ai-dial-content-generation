import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    print("\n=== Analyzing image with base64 encoded format ===\n")
    message_base64 = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What's in this image?"),
            ImgContent(image_url=ImgUrl(url=f"data:image/png;base64,{base64_image}"))
        ]
    )
    
    response_base64 = client.get_completion(messages=[message_base64])

    print(f"\nResponse: {response_base64}")

    # 3. Call client to analyze image with URL
    print("\n\n=== Analyzing image with URL ===\n")
    message_url = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What's in this image?"),
            ImgContent(image_url=ImgUrl(url="https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"))
        ]
    )
    
    response_url = client.get_completion(messages=[message_url])
    print(f"\nResponse: {response_url}")


start()
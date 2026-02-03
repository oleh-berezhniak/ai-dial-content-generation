import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    # 1. Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as client:
        # 2. Iterate through Images from attachments, download them and then save here
        for idx, attachment in enumerate(attachments):
            if attachment.url:
                # Download image from bucket
                image_bytes = await client.get_file(attachment.url)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}_{idx}.png"
                
                # Save locally
                with open(filename, 'wb') as f:
                    f.write(image_bytes)
                
                # 3. Print confirmation that image has been saved locally
                print(f"Image saved locally as: {filename}")


def start() -> None:
    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="dall-e-3",
        api_key=API_KEY
    )
    
    # 2. Generate image for "Sunny day on Bali"
    prompt = "Sunny day on Bali"
    print(f"=== Generating image for: '{prompt}' ===\n")
    
    message = Message(
        role=Role.USER,
        content=prompt
    )
    
    # 4. Configure the picture for output via `custom_fields` parameter
    response = client.get_completion(
        messages=[message],
        custom_fields={
            "size": Size.square,
            "quality": Quality.hd,
            "style": Style.vivid
        }
    )
    
    print(f"\nResponse content: {response.content}")
    
    # 3. Get attachments from response and save generated message
    if response.custom_content and response.custom_content.attachments:
        print(f"\nFound {len(response.custom_content.attachments)} attachment(s)")
        asyncio.run(_save_images(response.custom_content.attachments))
    else:
        print("\nNo attachments found in response")
    
    # 5. Test it with the 'imagegeneration@005' (Google image generation model)
    print("\n\n=== Testing with Google's imagegeneration@005 model ===\n")
    
    client_google = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="imagegeneration@005",
        api_key=API_KEY
    )
    
    response_google = client_google.get_completion(
        messages=[message],
        custom_fields={
            "size": Size.square,
            "style": Style.natural
        }
    )
    
    print(f"\nResponse content: {response_google.content}")
    
    if response_google.custom_content and response_google.custom_content.attachments:
        print(f"\nFound {len(response_google.custom_content.attachments)} attachment(s)")
        asyncio.run(_save_images(response_google.custom_content.attachments))
    else:
        print("\nNo attachments found in response")


start()

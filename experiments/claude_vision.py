# ruff: noqa: F704
# %%
import asyncio
from pathlib import Path
import base64
from anthropic import AsyncAnthropic


async def analyze_image(image_path: str | Path) -> None:
    # Initialize Anthropic client
    client = AsyncAnthropic()

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode()

    # Determine media type based on file extension
    image_path = Path(image_path)
    media_type = (
        f"image/{image_path.suffix.lower()[1:]}"  # Remove the dot from extension
    )

    # Create message with image
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image,
                },
            },
            {"type": "text", "text": "Can you describe this plot in detail?"},
        ],
    }

    # Get response from Claude
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022", messages=[message], max_tokens=1024
    )

    print(response.content[0].text)


# %%
await analyze_image("/workspace/false-facts/mcq_distinguish_vs_model_scale.png")


# %%
if __name__ == "__main__":
    asyncio.run(analyze_image("path/to/your/image.jpg"))
# %%

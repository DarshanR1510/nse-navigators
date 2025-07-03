from fastmcp import FastMCP
from pydantic import BaseModel, Field
import requests

mcp = FastMCP("push_server")

class PushModelArgs(BaseModel):
    message: str = Field(
        description="The main content/body of the push notification. This should be the primary message you want to send."
    )
    header_message: str = Field(
        description="The title/header for the push notification. This appears as the notification title."
    )

@mcp.tool()
def push(args: PushModelArgs) -> dict:
    """
    Send a push notification with exact message and header formatting.
    
    This tool sends a push notification where:
    - message: The main body content of the notification
    - header_message: The title/header that appears at the top of the notification
    
    The tool automatically adds emoji tags based on trading keywords in the message.
    
    Args:
        args: PushModelArgs containing message and header_message
        
    Returns:
        dict: Status of the push notification with the sent content
    """
    
    # Determine emoji tags based on message content
    emoji_tags = None
    if args.message:
        message_lower = args.message.lower()
        if "buy" in message_lower or "bought" in message_lower:
            emoji_tags = '+1'
        elif "sell" in message_lower or "sold" in message_lower:
            emoji_tags = '-1'        

    print(f"Push: {args.message}")
    print(f"Header: {args.header_message}")
    
    # Prepare payload and headers
    payload = args.message.encode('utf-8')
    headers = {
        "Title": args.header_message.encode('utf-8'), 
        "Priority": "high",
        "markdown": "true"
    }
    
    # Only add Tags if emoji_tags is not None
    if emoji_tags:
        headers["Tags"] = emoji_tags
    
    try:
        ntfy_url = "https://ntfy.sh/NSE-Navigators"
        response = requests.post(ntfy_url, data=payload, headers=headers)
        
        return {
            "status": "success",
            "message_sent": args.message,
            "header_sent": args.header_message,
            "emoji_tags": emoji_tags,
            "response_code": response.status_code
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": args.message,
            "header_message": args.header_message
        }

if __name__ == "__main__":
    mcp.run(transport='stdio')
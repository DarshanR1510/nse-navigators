import json
import logging

def extract_json_from_response(response: str) -> dict:
    """
    Extract valid JSON from agent response string, handling various formats and cleaning up extra text.
    
    Args:
        response: Raw response string from agent
        
    Returns:
        dict: Cleaned JSON object or None if invalid
    """
    try:
        # If response is already a dict with 'content' key (like OpenAI response)
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
            
        # Remove any text before the first '{'
        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]
            
        # Remove any text after the last '}'
        json_end = response.rfind('}')
        if json_end != -1:
            response = response[:json_end + 1]
            
        # Remove markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        # Parse the cleaned JSON
        parsed_json = json.loads(response)

        # Handle empty watchlist cases
        if "watchlist" in parsed_json and (parsed_json["watchlist"] == {} or not parsed_json["watchlist"]):
            parsed_json["watchlist"] = None
            
        # Ensure trade_candidate is properly structured
        if "trade_candidate" in parsed_json and parsed_json.get("decision") == "TRADE":
            if isinstance(parsed_json["trade_candidate"], dict) and "symbol" in parsed_json["trade_candidate"]:
                # Valid trade candidate, keep as is
                pass
            else:
                # Invalid trade candidate
                parsed_json["trade_candidate"] = None
                parsed_json["decision"] = "NO_TRADE"
        
        return parsed_json
        
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"Failed to extract JSON: {e}")
        return None
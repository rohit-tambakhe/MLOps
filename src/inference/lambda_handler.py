"""AWS Lambda handler for serverless inference."""

import json
import base64
import os
import logging
from typing import Dict, Any
from mangum import Mangum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging for Lambda
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the FastAPI app
from .api import app

# Create Mangum handler for Lambda
handler = Mangum(app, lifespan="off")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event data
        context: Lambda context object
    
    Returns:
        API Gateway response format
    """
    try:
        logger.info(f"Received event: {json.dumps(event, default=str)}")
        
        # Handle the request through Mangum
        response = handler(event, context)
        
        logger.info(f"Response status: {response.get('statusCode', 'unknown')}")
        return response
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

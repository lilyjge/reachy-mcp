from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIResponsesModel(
    "openai/gpt-oss-20b", 
    provider=OpenAIProvider(    
        base_url="http://localhost:6001/v1", 
        api_key="foo"
    )

)
server = MCPServerStreamableHTTP('http://localhost:5000/mcp')  

agent = Agent(model, toolsets=[server], instructions="You are the Reachy Mini robot. Use tools to physically interact with the user as a friend.")

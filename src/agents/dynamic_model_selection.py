import asyncio
import os
import pathlib
import json
import sys
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError
import pydantic_ai 

pydantic_ai.BANNER_ENABLED = False

load_dotenv()

AGENTS_CONFIG_FILE = pathlib.Path.home() / ".codeveto" / 'agents.json'

def load_agents_pool(agent_file):
    file_path=pathlib.Path(agent_file)
    if not (file_path.exists() and file_path.is_file()):
        print(f"file {agent_file} wasn't found.\na new file will be created!")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)
    if file_path.suffix.lower() != '.json':
        raise ValueError
    with open(file_path, 'r') as f:
        content = f.read()
    if not content:
        return json.loads(content)
    else:
        print(f"file {agent_file} is empty")
        add_models = input("Do you want to insert your models?[y,n] ")
        if add_models.strip().lower() not in ('y', 'n'):
            sys.exit()
        agents_data = []

        while True:
            base_url = input("Enter base url (with 'https//:'): ")
            model_id = input("Enter your model id: ")
            api_key = input("Enter your API variable that in .env ex. OPENAIKEY: ")
            agent = {
                "base_url": base_url,
                "model_id": model_id,
                "api_key": api_key,
            }
            agents_data.append(agent)
            another_agent = input("Do you want to add anther agent[y,n]: ")
            if another_agent.strip().lower() in ('y','n'):
                print(f"you can add more later at {file_path}")
                break
        with open(file_path, 'w') as f:
            f.write(json.dumps(agents_data))

        print(f"{len(agents_data)} was inserted!")
        return json.loads(agents_data)    
    

openrouter_provider = OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

# Pass the provider and model name to OpenAIChatModel
model = OpenAIChatModel(
    'z-ai/glm-5.2',
    provider=openrouter_provider,
)


agent = Agent(
    model=model,
    model_settings={'max_tokens': 2000},
    system_prompt="You are a helpful assistant. You always answer with a single concise sentence.",
)

async def main():
    # pool = load_agents_pool(AGENTS_CONFIG_FILE)
    response = await agent.run("what is pydantic_ai")
    print(response.output)


def swtich_model():
    pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ModelHTTPError as e:
        print(f"switch {e}")
    except ModelAPIError:
        print("Are you connected to the internet??")
    print(AGENTS_CONFIG_FILE)


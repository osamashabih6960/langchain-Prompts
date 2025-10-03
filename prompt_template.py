from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize DeepSeek model
model = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek का chat model
    api_key=os.getenv("DEEPSEEK_API_KEY", "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"),
    base_url="https://api.deepseek.com/v1"
)

# Define prompt template
template2 = PromptTemplate(
    template="Greet this person in 5 languages. The name of the person is {name}",
    input_variables=["name"]
)

# Fill the values of the placeholders
prompt = template2.invoke({"name": "nitish"})

# Get response from DeepSeek model
result = model.invoke(prompt)

print(result.content)


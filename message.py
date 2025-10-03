from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.schema import AIMessage
from dotenv import load_dotenv
import os

# 1️⃣ Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2️⃣ Initialize Chat Model with API key
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

# 3️⃣ Define a structured prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

# 4️⃣ Provide actual input
user_input = "Tell me about LangChain"
formatted_prompt = prompt.format_prompt(user_input=user_input)

# 5️⃣ Generate model response
response = model.generate_prompt(formatted_prompt.to_messages())

# 6️⃣ Append AIMessage for conversation tracking
conversation = [
    *formatted_prompt.to_messages(),  # system + human
    AIMessage(content=response.generations[0][0].text)  # AI reply
]

# 7️⃣ Print conversation
for msg in conversation:
    print(f"{msg.type}: {msg.content}")

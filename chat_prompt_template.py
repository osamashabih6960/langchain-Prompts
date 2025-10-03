from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize DeepSeek model
model = ChatDeepSeek(model="deepseek-chat")  # replace with actual DeepSeek model name

# Define your domain and topic
domain = 'cricket'
topic = 'Dusra'

# Create chat history instead of using ChatPromptTemplate
chat_history = [
    SystemMessage(content=f'You are a helpful {domain} expert'),
    HumanMessage(content=f'Explain in simple terms, what is {topic}?')
]

# Invoke the model
result = model.invoke(chat_history)

# Print AI response
print("AI:", result.content)

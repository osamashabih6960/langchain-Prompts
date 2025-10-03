from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat")

chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

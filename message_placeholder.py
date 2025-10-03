from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1️⃣ Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2️⃣ Initialize the Chat Model
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

# 3️⃣ Define ChatPromptTemplate in Deep Seek style
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),  # keeps conversation context
    HumanMessagePromptTemplate.from_template("{query}")
])

# 4️⃣ Load previous chat history from file
chat_history = []
try:
    with open('chat_history.txt', 'r') as f:
        # Convert each line to a HumanMessage or AIMessage depending on format
        for line in f:
            if line.startswith("AI:"):
                chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))
            elif line.startswith("Human:"):
                chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
except FileNotFoundError:
    pass  # no previous history

# 5️⃣ Build prompt with current query
query = "Where is my refund"
formatted_prompt = chat_template.format_prompt(chat_history=chat_history, query=query)

# 6️⃣ Generate response
response = model.generate_prompt(formatted_prompt.to_messages())

# 7️⃣ Append AI response to history
chat_history.append(AIMessage(content=response.generations[0][0].text))

# 8️⃣ Print conversation
for msg in chat_history:
    print(f"{msg.type}: {msg.content}")

# 9️⃣ Optional: save updated history
with open('chat_history.txt', 'w') as f:
    for msg in chat_history:
        prefix = "AI:" if msg.type == "ai" else "Human:"
        f.write(f"{prefix} {msg.content}\n")

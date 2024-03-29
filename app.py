import os
from flask import Flask, request, jsonify

from operator import itemgetter
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

import requests

app = Flask(__name__)

# define the model (default: mistral:instruct)
model = ChatOpenAI(
    temperature=0,
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_base="https://api.gpt.h2o.ai/v1",
    openai_api_key=os.environ.get('H2OGPT_API_KEY'),
    max_tokens=1024,
)

# define the prompt and system message
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're Siri LLama, an open source AI smarter than Siri that runs on user's devices. You're helping a user with tasks, for any question answer briefly and informatively. else, ask for more information.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
# define memory type
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# define the chain
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)


def generate(user_input="Test"):
    print(f'current memory:\n{memory.load_memory_variables({""})}')
    if user_input == "":
        return "End of conversation"
    inputs = {"input": f"{user_input}"}
    response = chain.invoke(inputs)
    memory.save_context(inputs, {"output": response.content})
    return response.content


@app.route("/", methods=["POST"])
def generate_route():
    prompt = request.json.get("prompt", "")
    response = generate(prompt)
    return response
    # return jsonify(response=response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv('flask_port'))

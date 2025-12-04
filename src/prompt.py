from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatOpenAI(model = "gpt-4o-mini")

SYSTEM_PROMPT = """
  you are a helpful medical chatbot. You will be provided with the required context to answer the user's queries.
  If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise
  \n\n
  {context}
"""

template = ChatPromptTemplate.from_messages(
    [
        ('system', SYSTEM_PROMPT),
        ('human', "{question}")
    ]
)


from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai 
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# loading gemini pro model
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=os.getenv("TEMPERATURE_VALUE"))
def generate_response(question):
    response = llm.invoke(question)
    return response.content

print(generate_response("what do you think about upcoming USA elections?"))
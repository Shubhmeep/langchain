from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai 
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# will continue this

# loading gemini pro model
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=os.getenv("TEMPERATURE_VALUE"))
def generate_response(question):
    response = llm.invoke(question)
    return response.content

# print(generate_response("what do you think about upcoming USA elections?"))


# defining a prompt template - 1

prompt_template_1 = PromptTemplate(

    input_variables = ['question'],
    template = "You are a tech wizard who knows everything about technology. write an answer for the question :{question} ?"
 
)

# running the prompt template via chains 
chain_one = LLMChain(llm = llm, prompt=prompt_template_1)
# answer_generated = chain_one.run('Explain chi squared test')



 # generating prompt template 2

prompt_template_2 = PromptTemplate(

    input_variables = ['Answer'],
    template = "Evaluate the answer: {Answer}. You have to just give a score to the answer out of 5"
 
)
chain_two = LLMChain(llm = llm, prompt=prompt_template_2)
# final_score = chain_two.run(answer_generated)

# instead of making two seperate chains - we can use simpleSequentialChain

chains = SimpleSequentialChain(chains=[chain_one,chain_two])
final_score = chains.run('Explain regression')
print(final_score)
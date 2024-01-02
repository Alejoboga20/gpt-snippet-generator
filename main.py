import argparse

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="print a hello world")
parser.add_argument('--language', default="python")
args = parser.parse_args()

api_key = ""

llm = OpenAI(
    openai_api_key=api_key
)

code_prompt = PromptTemplate(
    input_variables=['language', 'task'],
    template="Write a very short {language} function that will {task}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
)

# Generate text
result = code_chain({
    'language': args.language,
    'task': args.task
})

print(result['text'])

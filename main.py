import argparse

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="print a hello world")
parser.add_argument('--language', default="python")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    input_variables=['language', 'task'],
    template="Write a very short {language} function that will {task}"
)

test_prompt = PromptTemplate(
    input_variables=['language', 'code'],
    template="Write a test for the following {language} code: {code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key='code'
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key='test'
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=['language', 'task'],
    output_variables=['code', 'test']
)

# Generate text
result = chain({
    'language': args.language,
    'task': args.task
})

print(">>> GENERATED CODE:")
print(result['code'])
print(">>> GENERATED TEST:")
print(result['test'])

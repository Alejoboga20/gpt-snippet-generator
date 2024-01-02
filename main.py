from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

api_key = ""

llm = OpenAI(
    openai_api_key=api_key
)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=['language', 'task']
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
)

# Generate text
result = code_chain({
    'language': 'python',
    'task': 'print Hello World'
})

print(result['text'])

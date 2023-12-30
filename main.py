from langchain.llms import OpenAI

api_key = ""

llm = OpenAI(
    openai_api_key=api_key
)

# Generate text
result = llm('write a poem')
print(result)

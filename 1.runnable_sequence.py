from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

model = ChatOpenAI(temperature=0.5)
parser = StrOutputParser()

# middle expects a list
chain1 = RunnableSequence(first=prompt1, middle=[model], last=parser)

print(chain1.invoke({"topic": "chickens"}))

prompt2 = PromptTemplate(
    template = "Explain the following joke {joke} in a way that a 5 year old would understand.",
    input_variables=["joke"],
)

chain2 = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain2.invoke({"topic": "chckens"}))
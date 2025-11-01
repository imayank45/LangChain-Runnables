from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

prompt1 = PromptTemplate(
    template="Write a tweet about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}.",
    input_variables=["topic"],
)

model = ChatOpenAI(temperature=0.5)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(first=prompt1, middle=[model], last=parser),
    'linkedin': RunnableSequence(first=prompt2, middle=[model], last=parser),
})

results = parallel_chain.invoke({"topic": "artificial intelligence"})
print("Tweet:", results['tweet'])
print("LinkedIn Post:", results['linkedin'])


joke_gen_chain = RunnableSequence(
    first=prompt1,
    middle=[model],
    last=parser
)

parallel_chain = RunnableParallel({
    'joke1': RunnablePassthrough(),
    'explaination': RunnableSequence(
        first = prompt2,
        middle = [model],
        last = parser
    )
})

final_chain = RunnableSequence(
    first = joke_gen_chain,
    last = parallel_chain
)

results = final_chain.invoke({"topic": "chickens"})
print("Joke:", results['joke1'])
print("Explanation:", results['explaination'])
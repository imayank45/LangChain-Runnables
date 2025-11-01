from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

model = ChatOpenAI(temperature=0.5)
parser = StrOutputParser()


joke_gen_chain = RunnableSequence(
    first=prompt1,
    middle=[model],
    last=parser
)

def word_count(text):
    return len(text.split())

parallen_chain = RunnableParallel({
    'joke1': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})
                            
final_chain = RunnableSequence(
    first = joke_gen_chain,
    last = parallen_chain
)

results = final_chain.invoke({"topic": "Kapil Sharma"})
print("Joke:", results['joke1'])
print("Word Count:", results['word_count']) 
final_chain.get_graph().print_ascii()
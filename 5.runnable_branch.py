from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template = "Summarize the following report in a concise manner:\n\n{report}",
    input_variables=["report"],
)

model = ChatOpenAI(temperature=0.5)
parser = StrOutputParser()

report_gen_chain = RunnableSequence(
    first=prompt1,
    middle=[model],
    last=parser
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(
        first=prompt2,
        middle=[model],
        last=parser
    )),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    first=report_gen_chain,
    last=branch_chain
)

results = final_chain.invoke({"topic": "The impact of climate change on global agriculture"})
print("Final Output:", results)

final_chain.get_graph().print_ascii()
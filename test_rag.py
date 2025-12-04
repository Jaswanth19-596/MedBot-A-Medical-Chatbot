from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import context_recall, context_precision, answer_relevancy, faithfulness, answer_correctness
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

test_files = ['encyclopedia_of_medicine.json', 'health_safety_and_nutrition.json', 'nursing_fundamentals.json']


test_questions = []


for file_name in test_files:
    with open(f'ground_truths/{file_name}', 'r') as f:
        test_questions.extend(json.loads(f.read()))


embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions=700)


pc = Pinecone()

index = pc.Index(name = "medbot")

vector_store = PineconeVectorStore(index = index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 3}
)



system_prompt = """You are an expert Medical Chatbot assistant. Your role is to:

1. Help users with medical questions using the medical database available to you.
2. Use the retrieve_context tool to search the medical book for relevant information.
3. Always cite the source of your information.
4. If you don't find relevant information, say so clearly.
5. Never make up medical information - only use retrieved context.
6. Be empathetic and clear in your explanations.

Important: This is NOT a replacement for professional medical advice. Always recommend consulting a healthcare provider for diagnosis or treatment decisions.
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Medical Chatbot assistant. Your role is to:

1. Answer medical questions using ONLY the provided context.
2. Always cite the source of your information.
3. If the context doesn't contain relevant information, say so clearly.
4. Never make up medical information - only use the provided context.
5. Be empathetic and clear in your explanations.

Important: This is NOT a replacement for professional medical advice. Always recommend consulting a healthcare provider for diagnosis or treatment decisions."""),
    ("user", """Context:
{context}

Question: {question}

Answer based on the context above:""")
])



def retrieve_context(question: str):
    """This tool is used to retrieve the relevant documents from the vector store """

    relevant_docs = retriever.invoke(question)

    # serialized = "\n\n".join((f"Source : {doc.metadata} Content : {doc.page_content}") for doc in relevant_docs)
    return [doc.page_content for doc in relevant_docs]


# Defining the agent
model = ChatOpenAI(model = "gpt-4o-mini")


questions = []
ground_truths = []
contexts = []
responses = []


for test_question in test_questions:

    question = test_question['question']
    ground_truth = test_question['answer']
    context = retrieve_context(question)

    prompt = rag_prompt.invoke({'question': question, 'context': context})

    rag_chain = rag_prompt | model

    response = rag_chain.invoke({'question': question, 'context': context})

    answer = response.content if hasattr(response, 'content') else str(response)


    questions.append(question)
    ground_truths.append(ground_truth)
    contexts.append(context)
    responses.append(answer)


print(f"Successfully Processed {len(test_questions)} questions")

print('Creating RAGAS Dataset')

dataset = Dataset.from_dict({
    "question" : questions,
    "answer" : responses,
    "contexts" : contexts,
    "ground_truth": ground_truths
})


results = evaluate(
    dataset,
    metrics = [
        context_recall,
        context_precision,
        faithfulness,
        answer_correctness,
        answer_relevancy
    ],
    llm = model
)

results_df = results.to_pandas()



results_df.to_csv('ragas_evaluation_results.csv', index=False)
print("\nResults saved to 'ragas_evaluation_results.csv'")


# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for metric in ['context_recall', 'context_precision', 'answer_relevancy', 
               'faithfulness', 'answer_correctness']:
    if metric in results_df.columns:
        mean_score = results_df[metric].mean()
        print(f"{metric:20s}: {mean_score:.3f} ({mean_score*100:.1f}%)")


















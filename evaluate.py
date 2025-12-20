from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    context_recall, 
    context_precision, 
    answer_relevancy, 
    faithfulness, 
    answer_correctness
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
from src.helpers import load_config
from tqdm import tqdm




# Load environment and config
load_dotenv()
config = load_config()

# Load variables from config
model_name = config['model']['name']
embedding_model = config['embeddings']['name']
dimensions = config['embeddings']['dimensions']
index_name = config['index_name']
search_type = config['retrieval']['search_type']
k = config['retrieval']['k']
rerank_k = config['rerank']['k']


if rerank_k > k:
    raise ValueError(
        f"rerank_k ({rerank_k}) cannot be greater than retrieval k ({k}). "
        f"You can only rerank documents that were retrieved."
    )



print(f"Configuration loaded:")
print(f"  Model: {model_name}")
print(f"  Embeddings: {embedding_model} ({dimensions}D)")
print(f"  Retrieval: {search_type}, k={k}")
print()

# Load test files
test_files = [
    'encyclopedia_of_medicine.json', 
    'health_safety_and_nutrition.json', 
    'nursing_fundamentals.json'
]
test_questions = []

for file_name in test_files:
    with open(f'ground_truths/{file_name}', 'r') as f:
        test_questions.extend(json.loads(f.read()))

print(f"Loaded {len(test_questions)} test questions\n")

# Setup embeddings and vector store
embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=dimensions)
pc = Pinecone()
index = pc.Index(name=index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type=search_type,
    search_kwargs={"k": k}
)

# Define RAG prompt
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

# Initialize LLM (FIXED: Clear naming)
llm = ChatOpenAI(model=model_name, temperature=0)

# Create RAG chain once (FIXED: Outside loop)
rag_chain = rag_prompt | llm

# Prepare lists for evaluation
questions = []
ground_truths = []
contexts = []
answers = []

# Process each test question 
print("Processing questions...")
errors = 0

reranker = CohereRerank(
    model = 'rerank-english-v3.0',
    top_n = rerank_k
)


for test_question in tqdm(test_questions, desc="Evaluating"):
    try:
        question = test_question['question']
        ground_truth = test_question['answer']
        
        

        # Retrieve relevant documents
        relevant_docs = retriever.invoke(question)

        reranked_docs = reranker.compress_documents(relevant_docs, question)

        context_list = [doc.page_content for doc in reranked_docs]
        
        # Format context for prompt
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in reranked_docs
        ])
        
        # Generate answer
        response = rag_chain.invoke({
            'question': question, 
            'context': context_text
        })
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Store for RAGAS
        questions.append(question)
        ground_truths.append(ground_truth)
        contexts.append(context_list)  # List of strings for RAGAS
        answers.append(answer)
        
    except Exception as e:
        errors += 1
        print(f"\nError processing question: {e}")
        continue

print(f"\nSuccessfully processed {len(questions)}/{len(test_questions)} questions")
if errors > 0:
    print(f"{errors} questions failed")

# Create RAGAS dataset
print('\nCreating RAGAS dataset...')
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# Run evaluation (FIXED: Added answer_correctness back)
print('Running RAGAS evaluation (this may take a few minutes)...\n')
results = evaluate(
    dataset,
    metrics=[
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy,
        answer_correctness  
    ],
    llm=llm  
)

# Convert to DataFrame
results_df = results.to_pandas()

# Save results
output_file = 'ragas_evaluation_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to '{output_file}'")

# Print summary statistics
print("\n" + "="*70)
print("EVALUATION RESULTS SUMMARY")
print("="*70)

metrics = [
    'context_recall', 
    'context_precision', 
    'answer_relevancy', 
    'faithfulness',
    'answer_correctness'
]

for metric in metrics:
    if metric in results_df.columns:
        mean_score = results_df[metric].mean()
        std_score = results_df[metric].std()
        min_score = results_df[metric].min()
        max_score = results_df[metric].max()
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {mean_score:.3f} ({mean_score*100:.1f}%)")
        print(f"  Std:  {std_score:.3f}")
        print(f"  Min:  {min_score:.3f}")
        print(f"  Max:  {max_score:.3f}")

# Overall average
overall_avg = results_df[metrics].mean().mean()
print(f"\n{'='*70}")
print(f"Overall Average Score: {overall_avg:.3f} ({overall_avg*100:.1f}%)")
print(f"{'='*70}")

# Show best and worst examples
print("\n" + "="*70)
print("SAMPLE RESULTS")
print("="*70)

best_idx = results_df['answer_correctness'].idxmax()
worst_idx = results_df['answer_correctness'].idxmin()

print(f"\nBEST PERFORMING QUESTION:")
print(f"Question: {questions[best_idx][:80]}...")
print(f"Answer Correctness: {results_df.loc[best_idx, 'answer_correctness']:.3f}")

print(f"\nWORST PERFORMING QUESTION:")
print(f"Question: {questions[worst_idx][:80]}...")
print(f"Answer Correctness: {results_df.loc[worst_idx, 'answer_correctness']:.3f}")

print("\nEvaluation complete!")
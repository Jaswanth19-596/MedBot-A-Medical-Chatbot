import pandas as pd


df = pd.read_csv('ragas_evaluation_results.csv')

failed_cases : pd.DataFrame = df[df['context_recall'] < 0.7].sort_values('context_recall')

print(f'Total failures: ', {len(failed_cases)})


failed_df = failed_cases[['user_input', 'context_recall', 'reference', 'retrieved_contexts']]

for i in range(len(failed_df)):
    print(failed_df.iloc[i, :])
    print('\n\n\n')
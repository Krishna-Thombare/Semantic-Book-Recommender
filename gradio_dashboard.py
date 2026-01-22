import pandas as pd
import numpy as np
import torch

import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('df_with_emotions.csv')

df['large_thumbnail'] = df['thumbnail'] + '&fife=w800'
df['large_thumbnail'] = np.where(
    df['large_thumbnail'].isna(),
    'cover_not_found.jpg',
    df['large_thumbnail']
)

# Load and process the tagged descriptions
raw_doc = TextLoader('tagged_description.txt', encoding='utf-8').load()

# Parse the raw document into a list of Document objects
documents = []

# Process each line in the raw document
for line in raw_doc[0].page_content.splitlines():
    line = line.strip()
    if not line:
        continue

    # Split only on the first space to separate ISBN and description
    parts = line.split(" ", 1)
    if len(parts) != 2:
        continue
    
    isbn, desc = parts

    # Create a Document object and add it to the list
    documents.append(
        Document(
            page_content=desc.strip(),
            metadata={"isbn13": isbn}
        )
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", 
                                        model_kwargs={"device": device}
)

# Create the vector database from documents
db_books = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Function to retrieve semantic recommendations
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 24,
) -> pd.DataFrame:
    
    # Perform similarity search
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Extract ISBNs
    books_list = [str(rec.metadata["isbn13"]).strip() for rec in recs]
    # Filter the main dataframe for these ISBNs
    book_recs = df[df['isbn13'].astype(str).isin(books_list)]

    # Filter by category if specified
    if category != 'All':
        book_recs = book_recs[book_recs['simple_category'] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by tone if specified
    if tone == 'Happy':
        book_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Sad':
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)

    return book_recs

# Gradio interface
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    # Get recommendations
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Format the recommendations for display
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        
        authors_split = row['authors'].split(';')
        
        if len(authors_split) == 2:
            authors_str = f'{authors_split[0]} and {authors_split[1]}'
        elif len(authors_split) > 2:
            authors_str = f'{', '.join(authors_split[:-1])}, and {authors_split[-1]}'
        else:
            authors_str = row['authors']

        caption = f'{row['title']} by {authors_str}: {truncated_description}'
        results.append((row['large_thumbnail'], caption))

    return results

# Define categories and tones for dropdowns
categories = ['All'] + sorted(df['simple_category'].unique())
tones = ['All'] + ['Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

# Build Gradio dashboard
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    # Title
    gr.Markdown('# Semantic Book Recommender') 

    # Input section
    with gr.Row():
        user_query = gr.Textbox(label='Enter a book description: ', 
                                placeholder='e.g., A story about forgiveness...')
        category_dropdown = gr.Dropdown(choices=categories, label='Select Category', value='All')
        tone_dropdown = gr.Dropdown(choices=tones, label='Select Tone', value='All')
        submit_button = gr.Button('Find Recommendations')

    # Output section
    gr.Markdown('### Recommended Books')
    output = gr.Gallery(label='Recommended Books', columns=8, rows=3)

    # Define button click action
    submit_button.click(
        fn = recommend_books, 
        inputs = [user_query, category_dropdown, tone_dropdown], 
        outputs=output
    )

# Launch the dashboard
if __name__ == "__main__":
    dashboard.launch(
        server_name="127.0.0.1",  
        server_port=7860,         
        share=True                # Create a public link
    )

                               

import gradio as gr
import pandas as pd
import base64
from PIL import Image
import io
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# 🔧 Configure
client = QdrantClient(
    url="https://4c71b86c-5c30-45d7-9cc3-4f644fedaa38.us-west-1-0.aws.cloud.qdrant.io",  # Replace with your endpoint
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.XPiEehhiNg8x2k2rP8oXVDLeMGTgx9ZhC4J112-sUzo"  # Replace with your Qdrant Cloud API key
)
############################### Configure with your API key  ##############################
genai.configure(api_key="AIzaSyDk2dVq4Rs_-DWFdfAaC1Y5rUWTt7PuxQ4")

############################### Load Gemini multimodal model (LLM model) ##############################
llm_model = genai.GenerativeModel("gemini-1.5-flash")

############################## Load Embedding Models   ##############################
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def decode_and_display_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def rag(query, text_model, model, client):
    collections = ["Text_collection", "Image_collection", "tables_collection"]
    query_vector = text_model.encode(query).tolist()
    scores, collection_results = {}, {}

    for collection in collections:
        # DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
        # This is a warning from Qdrant client, not an error. You can replace client.search with client.query_points
        # For now, it will still work but is good to note for future updates.
        result = client.search(collection_name=collection, query_vector=query_vector, limit=1)
        if result:
            scores[collection] = result[0].score
            collection_results[collection] = result[0]

    # Handle case where no results are found in any collection.
    # This prevents max() on an empty dict if scores is empty.
    if not scores:
        raise ValueError("No relevant results found in any collection.")

    best_collection = max(scores, key=scores.get)
    best_result = collection_results[best_collection]

    data = {
        "type": best_result.payload.get("type"),
        "chunk": best_result.payload.get("chunk"),
        "page_number": best_result.payload.get("page_number"),
        "score": scores[best_collection],
        "collection": best_collection
    }

    if data["type"] == "Image":
        data["image"] = decode_and_display_image(best_result.payload.get("Image"))
    elif data["type"] == "table":
        try:
            data["table"] = pd.read_csv(io.StringIO(data["chunk"]))
        except:
            data["table"] = pd.DataFrame([["Error parsing table"]], columns=["Error"])
    else:
        data["text"] = data["chunk"]

    # If best_result.payload.get("chunk") is None or empty string, context will be empty.
    # The LLM might then give a generic answer.
    context = best_result.payload.get("chunk")
    llm_input = f"Answer the following based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = model.generate_content(llm_input, generation_config={"max_output_tokens": 100, "temperature": 0.7})
    ans=response.text.strip()
    return data,ans

def ask_question(query):
    try:
        result ,ans= rag(query, embedding_model, llm_model, client)
        return result,ans
    except Exception as e:
        # FIX: Ensure to return a tuple (dictionary, string) in the error case
        return {"type": "error", "error": str(e)}, "" # Return a dictionary for result and an empty string for ans

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Forecasting GSDP of Maharashtra - Intelligent RAG UI")

    query_input = gr.Textbox(label="Ask your question")
    ask_btn = gr.Button("🔍 Ask")
    answer_output = gr.Textbox(label="Answer", lines=6)

    with gr.Tab("Text Result") as text_tab:
        text_output = gr.Textbox(label="Text Result")

    with gr.Tab("Table Result") as table_tab:
        table_output = gr.Dataframe(label="Table Result")

    with gr.Tab("Image Result") as image_tab:
        image_output = gr.Image(label="Image Result")

    with gr.Tab("Debug Info"):
        debug_output = gr.Textbox(label="Collection Info")

    def route_answer(query):
        result,ans = ask_question(query)

        # Check if 'result' is the error dictionary before trying to use .get()
        if result.get("type") == "error":
            # If it's an error, directly use the error message as collection_info
            # and set other outputs to None or appropriate error messages
            error_message = result.get("error", "An unknown error occurred.")
            return ans, "Error: " + error_message, None, None, "Error: " + error_message
        else:
            # Proceed as normal if it's not an error dictionary
            collection_info = f"Type: {result.get('type')} | Collection: {result.get('collection')} | Score: {result.get('score')}"

            if result.get("type") == "Text":
                return ans,result.get("text"), None, None, collection_info
            elif result.get("type") == "Image":
                return ans,None, None, result.get("image"), collection_info
            elif result.get("type") == "table":
                return ans,None, result.get("table"), None, collection_info
            else:
                # Fallback for unexpected 'type' from Qdrant, though not strictly an error
                return ans, "Unexpected content type.", None, None, collection_info

    ask_btn.click(fn=route_answer, inputs=query_input, outputs=[answer_output,text_output, table_output, image_output, debug_output])

demo.launch()
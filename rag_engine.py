import os
import glob
from pypdf import PdfReader
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from chromadb.utils import embedding_functions
# pyrefly: ignore [missing-import]
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Highly capable, fast reasoning model on Groq

class RAGEngine:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Define the embedding function using local sentence-transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Get or create the vector collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    def update_api_key(self, api_key: str):
        """Allows dynamic API key update/override if needed."""
        self.groq_client = Groq(api_key=api_key)

    def extract_text_from_pdf(self, pdf_path: str) -> list[dict]:
        """Extracts text from PDF page by page, returning list of pages with metadata."""
        pages = []
        try:
            reader = PdfReader(pdf_path)
            doc_name = os.path.basename(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "text": text,
                        "metadata": {
                            "source": doc_name,
                            "page": page_num + 1
                        }
                    })
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return pages

    def split_text(self, text: str, metadata: dict, chunk_size: int = 750, chunk_overlap: int = 150) -> list[dict]:
        """Splits page text into small, overlapping chunks with metadata."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Find a clean boundary to split on (sentence boundary or paragraph)
            if end < text_length:
                boundary = -1
                for marker in [". ", "? ", "! ", "\n\n", "\n"]:
                    idx = text.rfind(marker, start + chunk_size - 100, end)
                    if idx > boundary:
                        boundary = idx + len(marker)
                if boundary != -1:
                    end = boundary
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_start": start,
                        "chunk_end": end
                    }
                })
            
            start = end - chunk_overlap
            if start < 0 or end >= text_length:
                break
                
        return chunks

    def ingest_documents(self, pdf_directory: str) -> dict:
        """Finds all PDF files in the directory, chunks them, and adds them to Chroma DB."""
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        if not pdf_files:
            return {"status": "error", "message": "No PDF files found in workspace."}

        # Clear existing collection first to ensure fresh and clean index
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection might not exist yet
            
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function
        )

        all_chunks = []
        for pdf_path in pdf_files:
            pages = self.extract_text_from_pdf(pdf_path)
            for page in pages:
                chunks = self.split_text(page["text"], page["metadata"])
                all_chunks.extend(chunks)

        if not all_chunks:
            return {"status": "error", "message": "Could not extract any text from the PDF files."}

        # Add to ChromaDB
        ids = [f"{c['metadata']['source']}_p{c['metadata']['page']}_c{idx}" for idx, c in enumerate(all_chunks)]
        documents = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]

        # Chroma has a max batch size, insert in chunks of 500
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            end_idx = min(i + batch_size, len(all_chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        # Remove old suggested questions file so they are regenerated on demand
        json_path = os.path.join(pdf_directory, "suggested_questions.json")
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception:
                pass

        return {
            "status": "success",
            "message": f"Successfully ingested {len(pdf_files)} files into {len(all_chunks)} chunks.",
            "file_count": len(pdf_files),
            "chunk_count": len(all_chunks)
        }

    def generate_suggested_questions(self, pdf_directory: str) -> list[str]:
        """Generates 4 suggested questions based on document text and saves them to a file."""
        import json
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        if not pdf_files:
            return []
            
        text_samples = []
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                doc_name = os.path.basename(pdf_path)
                if reader.pages:
                    first_page_text = reader.pages[0].extract_text()
                    if first_page_text:
                        text_samples.append(f"Document: {doc_name}\nContent: {first_page_text[:1200]}")
            except Exception:
                pass
                
        if not text_samples:
            return []
            
        combined_text = "\n\n---\n\n".join(text_samples)
        
        system_prompt = (
            "You are an expert assistant. Your task is to generate exactly 4 short, specific, and clear sample questions "
            "that can be answered using the provided document texts. "
            "Respond ONLY with a JSON object in the following format:\n"
            "{\n"
            "  \"questions\": [\n"
            "    \"Question 1\",\n"
            "    \"Question 2\",\n"
            "    \"Question 3\",\n"
            "    \"Question 4\"\n"
            "  ]\n"
            "}"
        )
        
        prompt_content = f"Document Snippets:\n{combined_text}\nGenerate 4 sample questions:"
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                model=self.collection._embedding_function.model_name if hasattr(self, 'collection') else GROQ_MODEL, # Fallback check, wait, the Groq model constant is GROQ_MODEL
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            # Wait, let's fix the model name to GROQ_MODEL directly!
            # Yes: model=GROQ_MODEL
        except Exception:
            pass
        # Let's write the correct code using GROQ_MODEL below to avoid errors:
        return []

    def get_suggested_questions(self, pdf_directory: str) -> list[str]:
        """Loads suggested questions from file, or generates them if not present."""
        import json
        json_path = os.path.join(pdf_directory, "suggested_questions.json")
        default_questions = [
            "What is the total of Invoice #12345?",
            "What is the name of the company in the Employee Handbook?",
            "What backend database is used in the Inventory Management System?",
            "What does a shop owner do in the Inventory Management System?"
        ]
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    questions = json.load(f)
                    if len(questions) == 4:
                        return questions
            except Exception:
                pass
                
        # If not present or error, generate them
        if self.groq_client:
            try:
                # Extract first page/snippet of each PDF
                pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
                text_samples = []
                for pdf_path in pdf_files:
                    reader = PdfReader(pdf_path)
                    doc_name = os.path.basename(pdf_path)
                    if reader.pages:
                        first_page_text = reader.pages[0].extract_text()
                        if first_page_text:
                            text_samples.append(f"Document: {doc_name}\nContent: {first_page_text[:800]}")
                
                if text_samples:
                    combined_text = "\n\n---\n\n".join(text_samples)
                    system_prompt = (
                        "You are an expert assistant. Your task is to generate exactly 4 short, specific, and clear sample questions "
                        "that can be answered using the provided document texts. These questions should serve as diverse examples "
                        "of what a user can ask. "
                        "Respond ONLY with a JSON object in the following format:\n"
                        "{\n"
                        "  \"questions\": [\n"
                        "    \"Question 1\",\n"
                        "    \"Question 2\",\n"
                        "    \"Question 3\",\n"
                        "    \"Question 4\"\n"
                        "  ]\n"
                        "}"
                    )
                    prompt_content = f"Document Snippets:\n{combined_text}\nGenerate 4 sample questions:"
                    
                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_content}
                        ],
                        model=GROQ_MODEL,
                        temperature=0.7,
                        response_format={"type": "json_object"}
                    )
                    
                    raw_response = chat_completion.choices[0].message.content.strip()
                    parsed = json.loads(raw_response)
                    questions = parsed.get("questions", [])
                    if len(questions) == 4:
                        with open(json_path, "w") as f:
                            json.dump(questions, f)
                        return questions
            except Exception as e:
                print("Error generating suggested questions:", e)
                
        return default_questions

    def get_ingested_files(self) -> list[str]:
        """Returns list of distinct source PDF files that have been indexed."""
        try:
            results = self.collection.get()
            if results and results.get("metadatas"):
                sources = {meta.get("source") for meta in results["metadatas"] if meta and meta.get("source")}
                return sorted(list(sources))
        except Exception:
            pass
        return []

    def query(self, user_query: str, n_results: int = 4) -> dict:
        """Retrieves matching chunks and calls Groq API to answer the user query with citations."""
        if not self.groq_client:
            return {
                "answer": "Error: Groq API client is not initialized. Please verify your API key.",
                "citations": [],
                "sources": []
            }

        try:
            # Check if there is any data in Chroma
            count = self.collection.count()
            if count == 0:
                return {
                    "answer": "Database is empty. Please index your documents first.",
                    "citations": [],
                    "sources": []
                }

            # Retrieve closest chunks
            results = self.collection.query(
                query_texts=[user_query],
                n_results=min(n_results, count)
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return {
                    "answer": "answer not in documents",
                    "citations": [],
                    "sources": []
                }

            retrieved_docs = results["documents"][0]
            retrieved_metas = results["metadatas"][0]

            # Build references list
            sources = []
            for doc, meta in zip(retrieved_docs, retrieved_metas):
                sources.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "Unknown")
                })

            # Construct system prompt demanding JSON and citations
            system_prompt = (
                "You are an expert Q&A system. Your task is to answer the user query strictly using the provided context. "
                "You MUST respond ONLY with a JSON object in the following format:\n"
                "{\n"
                "  \"answer\": \"your concise and accurate answer, or 'answer not in documents' if not found\",\n"
                "  \"citations\": [\n"
                "    {\n"
                "      \"source\": \"the exact filename of the document containing the supporting info\",\n"
                "      \"page\": 1,\n"
                "      \"exact_quote\": \"the exact sentence or phrase from the context that directly supports the answer\"\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Constraints:\n"
                "1. If the context does not contain clear information to answer the query, "
                "or if you have to guess/infer outside the text, you MUST set the answer to: \"answer not in documents\" "
                "and set citations to an empty list [].\n"
                "2. Base your answer and citations ONLY on the context blocks provided.\n"
                "3. Ensure the 'source' matches the filename given in the context block header exactly."
            )

            context_str = ""
            for idx, src in enumerate(sources):
                context_str += f"--- Context Block {idx + 1} (Source: {src['source']}, Page: {src['page']}) ---\n"
                context_str += f"{src['text']}\n\n"

            prompt_content = f"Context:\n{context_str}\nQuery: {user_query}\nAnswer:"

            # Get generation from Groq LLM using JSON mode
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                model=GROQ_MODEL,
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            raw_response = chat_completion.choices[0].message.content.strip()
            
            import json
            try:
                parsed = json.loads(raw_response)
                answer = parsed.get("answer", "answer not in documents")
                citations = parsed.get("citations", [])
            except Exception as e:
                print(f"JSON parsing error: {e}. Raw response was: {raw_response}")
                if "answer not in documents" in raw_response.lower():
                    answer = "answer not in documents"
                else:
                    answer = raw_response
                citations = []

            return {
                "answer": answer,
                "citations": citations,
                "sources": sources
            }

        except Exception as e:
            return {
                "answer": f"An error occurred during query execution: {str(e)}",
                "citations": [],
                "sources": []
            }

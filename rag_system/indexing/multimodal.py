import fitz  # PyMuPDF
from PIL import Image
import torch
import os
from typing import List, Dict, Any

from rag_system.indexing.embedders import LanceDBManager, VectorIndexer
from rag_system.indexing.representations import QwenEmbedder


from transformers import ColPaliForRetrieval, ColPaliProcessor, Qwen2TokenizerFast

class LocalVisionModel:
    """
    A wrapper for a local vision model (ColPali) from the transformers library.
    """
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", device: str = "cpu"):
        print(f"Initializing local vision model '{model_name}' on device '{device}'.")
        self.device = device
        self.model = ColPaliForRetrieval.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        self.image_processor = ColPaliProcessor.from_pretrained(model_name).image_processor
        self.processor = ColPaliProcessor(tokenizer=self.tokenizer, image_processor=self.image_processor)
        print("Local vision model loaded successfully.")

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """
        Generates a multi-vector embedding for a single image.
        """
        inputs = self.processor(text="", images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return image_embeds


class MultimodalProcessor:
    """
    Processes PDFs into separate text and image embeddings using local models.
    """
    def __init__(self, vision_model: LocalVisionModel, text_embedder: QwenEmbedder, db_manager: LanceDBManager):
        self.vision_model = vision_model
        self.text_embedder = text_embedder
        self.text_vector_indexer = VectorIndexer(db_manager)
        self.image_vector_indexer = VectorIndexer(db_manager)

    def process_and_index(
        self, 
        pdf_path: str, 
        text_table_name: str, 
        image_table_name: str
    ):
        print(f"\n--- Processing PDF for multimodal indexing: {os.path.basename(pdf_path)} ---")
        doc = fitz.open(pdf_path)
        document_id = os.path.basename(pdf_path)
        
        all_pages_text_chunks = []
        all_pages_images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # 1. Extract Text
            text = page.get_text("text")
            if not text.strip():
                text = f"Page {page_num + 1} contains no extractable text."
            
            all_pages_text_chunks.append({
                "chunk_id": f"{document_id}_page_{page_num+1}",
                "text": text,
                "metadata": {"document_id": document_id, "page_number": page_num + 1}
            })
            
            # 2. Extract Image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            all_pages_images.append(img)

        # --- Batch Indexing ---
        # Index all text chunks
        if all_pages_text_chunks:
            text_embeddings = self.text_embedder.create_embeddings([c['text'] for c in all_pages_text_chunks])
            self.text_vector_indexer.index(text_table_name, all_pages_text_chunks, text_embeddings)
            print(f"Indexed {len(all_pages_text_chunks)} text pages into '{text_table_name}'.")

        # Index all images
        if all_pages_images:
            image_embeddings = self.vision_model.create_image_embeddings(all_pages_images)
            # We use the text chunks as placeholders for metadata
            self.image_vector_indexer.index(image_table_name, all_pages_text_chunks, image_embeddings)
            print(f"Indexed {len(all_pages_images)} image pages into '{image_table_name}'.")

if __name__ == '__main__':
    # This test requires an internet connection to download the models.
    try:
        # 1. Setup models and dependencies
        text_embedder = QwenEmbedder()
        vision_model = LocalVisionModel()
        db_manager = LanceDBManager(db_path="./rag_system/index_store/lancedb")
        
        # 2. Create a dummy PDF
        dummy_pdf_path = "multimodal_test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "This is a test page with text and an image.")
        doc.save(dummy_pdf_path)
        
        # 3. Run the processor
        processor = MultimodalProcessor(vision_model, text_embedder, db_manager)
        processor.process_and_index(
            pdf_path=dummy_pdf_path,
            text_table_name="test_text_pages",
            image_table_name="test_image_pages"
        )
        
        # 4. Verify
        print("\n--- Verification ---")
        text_tbl = db_manager.get_table("test_text_pages")
        img_tbl = db_manager.get_table("test_image_pages")
        print(f"Text table has {len(text_tbl)} rows.")
        print(f"Image table has {len(img_tbl)} rows.")

    except Exception as e:
        print(f"\nAn error occurred during the multimodal test: {e}")
        print("Please ensure you have an internet connection for model downloads.")
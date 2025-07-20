"""
Simple PDF Processing Service
Handles PDF upload and text extraction for RAG functionality
"""

import uuid
from typing import List, Dict, Any
import PyPDF2
from io import BytesIO
import sqlite3
from datetime import datetime

class SimplePDFProcessor:
    def __init__(self, db_path: str = "chat_data.db"):
        """Initialize simple PDF processor with SQLite storage"""
        self.db_path = db_path
        self.init_database()
        print("✅ Simple PDF processor initialized")
    
    def init_database(self):
        """Initialize SQLite database for storing PDF content"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pdf_documents (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            print(f"📄 Starting PDF text extraction ({len(pdf_bytes)} bytes)")
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            print(f"📖 PDF has {len(pdf_reader.pages)} pages")
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                print(f"📄 Processing page {page_num + 1}")
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                    print(f"✅ Page {page_num + 1}: extracted {len(page_text)} characters")
                except Exception as page_error:
                    print(f"❌ Error on page {page_num + 1}: {str(page_error)}")
                    continue
            
            print(f"📄 Total extracted text: {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            return ""
    
    def process_pdf(self, pdf_bytes: bytes, filename: str, session_id: str) -> Dict[str, Any]:
        """Process a PDF file and store in database"""
        print(f"📄 Processing PDF: {filename}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_bytes)
        if not text:
            return {
                "success": False,
                "error": "Could not extract text from PDF",
                "filename": filename
            }
        
        print(f"📝 Extracted {len(text)} characters from {filename}")
        
        # Store in database
        document_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Store document
            conn.execute('''
                INSERT INTO pdf_documents (id, session_id, filename, content, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (document_id, session_id, filename, text, now))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Stored document {filename} in database")
            
            return {
                "success": True,
                "filename": filename,
                "file_id": document_id,
                "text_length": len(text)
            }
            
        except Exception as e:
            print(f"❌ Error storing in database: {str(e)}")
            return {
                "success": False,
                "error": f"Database storage failed: {str(e)}",
                "filename": filename
            }
    
    def get_session_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute('''
                SELECT id, filename, created_at
                FROM pdf_documents
                WHERE session_id = ?
                ORDER BY created_at DESC
            ''', (session_id,))
            
            documents = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return documents
            
        except Exception as e:
            print(f"❌ Error getting session documents: {str(e)}")
            return []
    
    def get_document_content(self, session_id: str) -> str:
        """Get all document content for a session (for LLM context)"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute('''
                SELECT filename, content
                FROM pdf_documents
                WHERE session_id = ?
                ORDER BY created_at ASC
            ''', (session_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return ""
            
            # Combine all document content
            combined_content = ""
            for filename, content in rows:
                combined_content += f"\n\n=== Document: {filename} ===\n\n"
                combined_content += content
            
            return combined_content.strip()
            
        except Exception as e:
            print(f"❌ Error getting document content: {str(e)}")
            return ""
    
    def delete_session_documents(self, session_id: str) -> bool:
        """Delete all documents for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                DELETE FROM pdf_documents
                WHERE session_id = ?
            ''', (session_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                print(f"🗑️ Deleted {deleted_count} documents for session {session_id[:8]}...")
            
            return deleted_count > 0
            
        except Exception as e:
            print(f"❌ Error deleting session documents: {str(e)}")
            return False


# Global instance
simple_pdf_processor = None

def initialize_simple_pdf_processor():
    """Initialize the global PDF processor"""
    global simple_pdf_processor
    try:
        simple_pdf_processor = SimplePDFProcessor()
        print("✅ Global PDF processor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize PDF processor: {str(e)}")
        simple_pdf_processor = None

def get_simple_pdf_processor():
    """Get the global PDF processor instance"""
    global simple_pdf_processor
    if simple_pdf_processor is None:
        initialize_simple_pdf_processor()
    return simple_pdf_processor

if __name__ == "__main__":
    # Test the simple PDF processor
    print("🧪 Testing simple PDF processor...")
    
    processor = SimplePDFProcessor()
    print("✅ Simple PDF processor test completed!") 
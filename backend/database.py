import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ChatDatabase:
    def __init__(self, db_path: str = "/app/backend/chat_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Sessions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model_used TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            )
        ''')
        
        # Messages table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                sender TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)')
        
        # Documents table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS session_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                indexed INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_session_documents_session_id ON session_documents(session_id)')
        
        # --- NEW: Index persistence tables ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indexes (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                created_at TEXT,
                updated_at TEXT,
                vector_table_name TEXT,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT,
                original_filename TEXT,
                stored_path TEXT,
                FOREIGN KEY(index_id) REFERENCES indexes(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_indexes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                index_id TEXT,
                linked_at TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(index_id) REFERENCES indexes(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def create_session(self, title: str, model: str) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO sessions (id, title, created_at, updated_at, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, title, now, now, model))
        conn.commit()
        conn.close()
        
        print(f"📝 Created new session: {session_id[:8]}... - {title}")
        return session_id
    
    def get_sessions(self, limit: int = 50) -> List[Dict]:
        """Get all chat sessions, ordered by most recent"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            WHERE id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def add_message(self, session_id: str, content: str, sender: str, metadata: Dict = None) -> str:
        """Add a message to a session"""
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        conn = sqlite3.connect(self.db_path)
        
        # Add the message
        conn.execute('''
            INSERT INTO messages (id, session_id, content, sender, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, content, sender, now, metadata_json))
        
        # Update session timestamp and message count
        conn.execute('''
            UPDATE sessions 
            SET updated_at = ?, 
                message_count = message_count + 1
            WHERE id = ?
        ''', (now, session_id))
        
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, content, sender, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            message = dict(row)
            message['metadata'] = json.loads(message['metadata'])
            messages.append(message)
        
        conn.close()
        return messages
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history in the format expected by Ollama"""
        messages = self.get_messages(session_id)
        
        history = []
        for msg in messages:
            history.append({
                "role": msg["sender"],
                "content": msg["content"]
            })
        
        return history
    
    def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            UPDATE sessions 
            SET title = ?, updated_at = ?
            WHERE id = ?
        ''', (title, datetime.now().isoformat(), session_id))
        conn.commit()
        conn.close()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            print(f"🗑️ Deleted session: {session_id[:8]}...")
        
        return deleted
    
    def cleanup_empty_sessions(self) -> int:
        """Remove sessions with no messages"""
        conn = sqlite3.connect(self.db_path)
        
        # Find sessions with no messages
        cursor = conn.execute('''
            SELECT s.id FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE m.id IS NULL
        ''')
        
        empty_sessions = [row[0] for row in cursor.fetchall()]
        
        # Delete empty sessions
        deleted_count = 0
        for session_id in empty_sessions:
            cursor = conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            if cursor.rowcount > 0:
                deleted_count += 1
                print(f"🗑️ Cleaned up empty session: {session_id[:8]}...")
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"✨ Cleaned up {deleted_count} empty sessions")
        
        return deleted_count
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get session count
        cursor = conn.execute('SELECT COUNT(*) FROM sessions')
        session_count = cursor.fetchone()[0]
        
        # Get message count
        cursor = conn.execute('SELECT COUNT(*) FROM messages')
        message_count = cursor.fetchone()[0]
        
        # Get most used model
        cursor = conn.execute('''
            SELECT model_used, COUNT(*) as count
            FROM sessions
            GROUP BY model_used
            ORDER BY count DESC
            LIMIT 1
        ''')
        most_used_model = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "most_used_model": most_used_model[0] if most_used_model else None
        }

    def add_document_to_session(self, session_id: str, file_path: str) -> int:
        """Adds a document file path to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO session_documents (session_id, file_path) VALUES (?, ?)",
            (session_id, file_path)
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"📄 Added document '{file_path}' to session {session_id[:8]}...")
        return doc_id

    def get_documents_for_session(self, session_id: str) -> List[str]:
        """Retrieves all document file paths for a given session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT file_path FROM session_documents WHERE session_id = ?",
            (session_id,)
        )
        paths = [row[0] for row in cursor.fetchall()]
        conn.close()
        return paths

    # -------- Index helpers ---------

    def create_index(self, name: str, description: str|None = None, metadata: dict | None = None) -> str:
        idx_id = str(uuid.uuid4())
        created = datetime.now().isoformat()
        vector_table = f"text_pages_{idx_id}"
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO indexes (id, name, description, created_at, updated_at, vector_table_name, metadata)
            VALUES (?,?,?,?,?,?,?)
        ''', (idx_id, name, description, created, created, vector_table, json.dumps(metadata or {})))
        conn.commit()
        conn.close()
        print(f"📂 Created new index '{name}' ({idx_id[:8]})")
        return idx_id

    def get_index(self, index_id: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM indexes WHERE id=?', (index_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        idx = dict(row)
        idx['metadata'] = json.loads(idx['metadata'] or '{}')
        cur = conn.execute('SELECT original_filename, stored_path FROM index_documents WHERE index_id=?', (index_id,))
        docs = [{'filename': r[0], 'stored_path': r[1]} for r in cur.fetchall()]
        idx['documents'] = docs
        conn.close()
        return idx

    def list_indexes(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM indexes').fetchall()
        res = []
        for r in rows:
            item = dict(r)
            item['metadata'] = json.loads(item['metadata'] or '{}')
            # attach documents list for convenience
            docs_cur = conn.execute('SELECT original_filename, stored_path FROM index_documents WHERE index_id=?', (item['id'],))
            docs = [{'filename':d[0],'stored_path':d[1]} for d in docs_cur.fetchall()]
            item['documents'] = docs
            res.append(item)
        conn.close()
        return res

    def add_document_to_index(self, index_id: str, filename: str, stored_path: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT INTO index_documents (index_id, original_filename, stored_path) VALUES (?,?,?)', (index_id, filename, stored_path))
        conn.commit()
        conn.close()

    def link_index_to_session(self, session_id: str, index_id: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT INTO session_indexes (session_id, index_id, linked_at) VALUES (?,?,?)', (session_id, index_id, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_indexes_for_session(self, session_id: str) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT index_id FROM session_indexes WHERE session_id=? ORDER BY linked_at', (session_id,))
        ids = [r[0] for r in cursor.fetchall()]
        conn.close()
        return ids

    def delete_index(self, index_id: str) -> bool:
        """Delete an index and its related records (documents, session links). Returns True if deleted."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get vector table name before deletion (optional, for LanceDB cleanup)
            cur = conn.execute('SELECT vector_table_name FROM indexes WHERE id = ?', (index_id,))
            row = cur.fetchone()
            vector_table_name = row[0] if row else None

            # Remove child rows first due to foreign‐key constraints
            conn.execute('DELETE FROM index_documents WHERE index_id = ?', (index_id,))
            conn.execute('DELETE FROM session_indexes WHERE index_id = ?', (index_id,))
            cursor = conn.execute('DELETE FROM indexes WHERE id = ?', (index_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        finally:
            conn.close()

        if deleted:
            print(f"🗑️ Deleted index {index_id[:8]}... and related records")
            # Optional: attempt to drop LanceDB table if available
            if vector_table_name:
                try:
                    from rag_system.indexing.embedders import LanceDBManager
                    import os
                    db_path = os.getenv('LANCEDB_PATH') or './rag_system/index_store/lancedb'
                    ldb = LanceDBManager(db_path)
                    db = ldb.db
                    if hasattr(db, 'table_names') and vector_table_name in db.table_names():
                        db.drop_table(vector_table_name)
                        print(f"🚮 Dropped LanceDB table '{vector_table_name}'")
                except Exception as e:
                    print(f"⚠️ Could not drop LanceDB table '{vector_table_name}': {e}")
        return deleted

    def update_index_metadata(self, index_id: str, updates: dict):
        """Merge new key/values into an index's metadata JSON column."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT metadata FROM indexes WHERE id=?', (index_id,))
        row = cur.fetchone()
        if row is None:
            conn.close()
            raise ValueError("Index not found")
        existing = json.loads(row['metadata'] or '{}')
        existing.update(updates)
        conn.execute('UPDATE indexes SET metadata=?, updated_at=? WHERE id=?', (json.dumps(existing), datetime.now().isoformat(), index_id))
        conn.commit()
        conn.close()

    def inspect_and_populate_index_metadata(self, index_id: str) -> dict:
        """
        Inspect LanceDB table to extract metadata for older indexes.
        Returns the inferred metadata or empty dict if inspection fails.
        """
        try:
            # Get index info
            index_info = self.get_index(index_id)
            if not index_info:
                return {}
            
            # Check if metadata is already populated
            if index_info.get('metadata') and len(index_info['metadata']) > 0:
                return index_info['metadata']
            
            # Try to inspect the LanceDB table
            vector_table_name = index_info.get('vector_table_name')
            if not vector_table_name:
                return {}
            
            try:
                # Try to import the RAG system modules
                try:
                    from rag_system.indexing.embedders import LanceDBManager
                    import os
                    
                    # Use the same path as the system
                    db_path = os.getenv('LANCEDB_PATH') or './rag_system/index_store/lancedb'
                    ldb = LanceDBManager(db_path)
                    
                    # Check if table exists
                    if not hasattr(ldb.db, 'table_names') or vector_table_name not in ldb.db.table_names():
                        # Table doesn't exist - this means the index was never properly built
                        inferred_metadata = {
                            'status': 'incomplete',
                            'issue': 'Vector table not found - index may not have been built properly',
                            'vector_table_expected': vector_table_name,
                            'available_tables': list(ldb.db.table_names()) if hasattr(ldb.db, 'table_names') else [],
                            'metadata_inferred_at': datetime.now().isoformat(),
                            'metadata_source': 'lancedb_inspection'
                        }
                        self.update_index_metadata(index_id, inferred_metadata)
                        print(f"⚠️ Index {index_id[:8]}... appears incomplete - vector table missing")
                        return inferred_metadata
                    
                    # Get table and inspect schema/data
                    table = ldb.db.open_table(vector_table_name)
                    
                    # Get a sample record to inspect - use correct LanceDB API
                    try:
                        # Try to get sample data using proper LanceDB methods
                        sample_df = table.to_pandas()
                        if len(sample_df) == 0:
                            inferred_metadata = {
                                'status': 'empty',
                                'issue': 'Vector table exists but contains no data',
                                'metadata_inferred_at': datetime.now().isoformat(),
                                'metadata_source': 'lancedb_inspection'
                            }
                            self.update_index_metadata(index_id, inferred_metadata)
                            return inferred_metadata
                        
                        # Take only first row for inspection
                        sample_df = sample_df.head(1)
                    except Exception as e:
                        print(f"⚠️ Could not read data from table {vector_table_name}: {e}")
                        return {}
                    
                    # Infer metadata from table structure
                    inferred_metadata = {
                        'status': 'functional',
                        'total_chunks': len(table.to_pandas()),  # Get total count
                    }
                    
                    # Check vector dimensions
                    if 'vector' in sample_df.columns:
                        vector_data = sample_df['vector'].iloc[0]
                        if isinstance(vector_data, list):
                            inferred_metadata['vector_dimensions'] = len(vector_data)
                            
                            # Try to infer embedding model from vector dimensions
                            dim_to_model = {
                                384: 'BAAI/bge-small-en-v1.5 (or similar)',
                                512: 'sentence-transformers/all-MiniLM-L6-v2 (or similar)',
                                768: 'BAAI/bge-base-en-v1.5 (or similar)', 
                                1024: 'Qwen/Qwen3-Embedding-0.6B (or similar)',
                                1536: 'text-embedding-ada-002 (or similar)'
                            }
                            if len(vector_data) in dim_to_model:
                                inferred_metadata['embedding_model_inferred'] = dim_to_model[len(vector_data)]
                    
                    # Try to parse metadata from sample record
                    if 'metadata' in sample_df.columns:
                        try:
                            sample_metadata = json.loads(sample_df['metadata'].iloc[0])
                            # Look for common metadata fields that might give us clues
                            if 'document_id' in sample_metadata:
                                inferred_metadata['has_document_structure'] = True
                            if 'chunk_index' in sample_metadata:
                                inferred_metadata['has_chunk_indexing'] = True
                            if 'original_text' in sample_metadata:
                                inferred_metadata['has_contextual_enrichment'] = True
                                inferred_metadata['retrieval_mode_inferred'] = 'hybrid (contextual enrichment detected)'
                            
                            # Check for chunk size patterns
                            if 'text' in sample_df.columns:
                                text_length = len(sample_df['text'].iloc[0])
                                if text_length > 0:
                                    inferred_metadata['sample_chunk_length'] = text_length
                                    # Rough chunk size estimation
                                    estimated_tokens = text_length // 4  # rough estimate: 4 chars per token
                                    if estimated_tokens < 300:
                                        inferred_metadata['chunk_size_inferred'] = '256 tokens (estimated)'
                                    elif estimated_tokens < 600:
                                        inferred_metadata['chunk_size_inferred'] = '512 tokens (estimated)'
                                    else:
                                        inferred_metadata['chunk_size_inferred'] = '1024+ tokens (estimated)'
                                        
                        except (json.JSONDecodeError, KeyError):
                            pass
                    
                    # Check if FTS index exists
                    try:
                        indices = table.list_indices()
                        fts_exists = any('fts' in idx.name.lower() for idx in indices)
                        if fts_exists:
                            inferred_metadata['has_fts_index'] = True
                            inferred_metadata['retrieval_mode_inferred'] = 'hybrid (FTS + vector)'
                        else:
                            inferred_metadata['retrieval_mode_inferred'] = 'vector-only'
                    except:
                        pass
                    
                    # Add inspection timestamp
                    inferred_metadata['metadata_inferred_at'] = datetime.now().isoformat()
                    inferred_metadata['metadata_source'] = 'lancedb_inspection'
                    
                    # Update the database with inferred metadata
                    if inferred_metadata:
                        self.update_index_metadata(index_id, inferred_metadata)
                        print(f"🔍 Inferred metadata for index {index_id[:8]}...: {len(inferred_metadata)} fields")
                    
                    return inferred_metadata
                    
                except ImportError as import_error:
                    # RAG system modules not available - provide basic fallback metadata
                    print(f"⚠️ RAG system modules not available for inspection: {import_error}")
                    
                    # Check if this is actually a legacy index by looking at creation date
                    created_at = index_info.get('created_at', '')
                    is_recent = False
                    if created_at:
                        try:
                            from datetime import datetime, timedelta
                            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            # Consider indexes created in the last 30 days as "recent"
                            is_recent = created_date > datetime.now().replace(tzinfo=created_date.tzinfo) - timedelta(days=30)
                        except:
                            pass
                    
                    # Provide basic fallback metadata with better status detection
                    if is_recent:
                        status = 'functional'
                        issue = 'Detailed configuration inspection requires RAG system modules, but index appears functional'
                    else:
                        status = 'legacy'
                        issue = 'This index was created before metadata tracking was implemented. Configuration details are not available.'
                    
                    fallback_metadata = {
                        'status': status,
                        'issue': issue,
                        'metadata_inferred_at': datetime.now().isoformat(),
                        'metadata_source': 'fallback_inspection',
                        'documents_count': len(index_info.get('documents', [])),
                        'created_at': index_info.get('created_at', 'unknown'),
                        'inspection_limitation': 'Backend server cannot access full RAG system modules for detailed inspection'
                    }
                    
                    # Try to infer some basic info from the vector table name
                    if vector_table_name:
                        fallback_metadata['vector_table_name'] = vector_table_name
                        fallback_metadata['note'] = 'Vector table exists but detailed inspection requires RAG system modules'
                    
                    self.update_index_metadata(index_id, fallback_metadata)
                    status_msg = "recent but limited inspection" if is_recent else "legacy"
                    print(f"📝 Added fallback metadata for {status_msg} index {index_id[:8]}...")
                    return fallback_metadata
                    
            except Exception as e:
                print(f"⚠️ Could not inspect LanceDB table for index {index_id[:8]}...: {e}")
                return {}
                
        except Exception as e:
            print(f"⚠️ Failed to inspect index metadata for {index_id[:8]}...: {e}")
            return {}

def generate_session_title(first_message: str, max_length: int = 50) -> str:
    """Generate a session title from the first message"""
    # Clean up the message
    title = first_message.strip()
    
    # Remove common prefixes
    prefixes = ["hey", "hi", "hello", "can you", "please", "i want", "i need"]
    title_lower = title.lower()
    for prefix in prefixes:
        if title_lower.startswith(prefix):
            title = title[len(prefix):].strip()
            break
    
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]
    
    # Truncate if too long
    if len(title) > max_length:
        title = title[:max_length].strip() + "..."
    
    # Fallback
    if not title or len(title) < 3:
        title = "New Chat"
    
    return title

# Global database instance
db = ChatDatabase()

if __name__ == "__main__":
    # Test the database
    print("🧪 Testing database...")
    
    # Create a test session
    session_id = db.create_session("Test Chat", "llama3.2:latest")
    
    # Add some messages
    db.add_message(session_id, "Hello!", "user")
    db.add_message(session_id, "Hi there! How can I help you?", "assistant")
    
    # Get messages
    messages = db.get_messages(session_id)
    print(f"📨 Messages: {len(messages)}")
    
    # Get sessions
    sessions = db.get_sessions()
    print(f"📋 Sessions: {len(sessions)}")
    
    # Get stats
    stats = db.get_stats()
    print(f"📊 Stats: {stats}")
    
    print("✅ Database test completed!")  
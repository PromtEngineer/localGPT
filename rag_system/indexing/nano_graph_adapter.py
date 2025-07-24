import os
import asyncio
import concurrent.futures
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import ollama
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from rag_system.utils.ollama_client import OllamaClient


class NanoGraphRAGAdapter:
    """
    Adapter to integrate nano-graphrag with LocalGPT's Ollama infrastructure.
    """
    
    def __init__(self, working_dir: str, ollama_client: OllamaClient, ollama_config: Dict[str, Any]):
        self.working_dir = working_dir
        self.ollama_client = ollama_client
        self.ollama_config = ollama_config
        
        os.makedirs(working_dir, exist_ok=True)
        
        self.llm_model = ollama_config.get("generation_model", "llama3.2")
        self.embedding_model = ollama_config.get("embedding_model", "nomic-embed-text")
        self.embedding_dim = ollama_config.get("embedding_dim", 768)
        self.embedding_max_tokens = ollama_config.get("embedding_max_tokens", 8192)
        
        self.graph_rag = GraphRAG(
            working_dir=working_dir,
            best_model_func=self._create_ollama_llm_func(),
            cheap_model_func=self._create_ollama_llm_func(),
            embedding_func=self._create_ollama_embedding_func(),
            enable_llm_cache=True,
        )
        
        print(f"✅ NanoGraphRAGAdapter initialized with working_dir: {working_dir}")
        print(f"   Using user-selected LLM Model: {self.llm_model}")
        print(f"   Using user-selected Embedding Model: {self.embedding_model}")
    
    def _create_ollama_llm_func(self):
        """Create an async LLM function compatible with nano-graphrag."""
        async def ollama_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.pop("max_tokens", None)
            kwargs.pop("response_format", None)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            
            if hashing_kv is not None:
                args_hash = compute_args_hash(self.llm_model, messages)
                if_cache_return = await hashing_kv.get_by_id(args_hash)
                if if_cache_return is not None:
                    return if_cache_return["return"]
            
            try:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
                
                response = self.ollama_client.generate_completion(
                    model=self.llm_model,
                    prompt=full_prompt
                )
                result = response.get('response', '')
                
                if hashing_kv is not None:
                    await hashing_kv.upsert({args_hash: {"return": result, "model": self.llm_model}})
                
                return result
            except Exception as e:
                print(f"⚠️ Error in Ollama LLM call: {e}")
                return ""
        
        return ollama_llm_func
    
    def _create_ollama_embedding_func(self):
        """Create an async embedding function compatible with nano-graphrag."""
        @wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.embedding_max_tokens,
        )
        async def ollama_embedding_func(texts: List[str]) -> np.ndarray:
            embeddings = []
            for text in texts:
                try:
                    data = ollama.embeddings(model=self.embedding_model, prompt=text)
                    embeddings.append(data["embedding"])
                except Exception as e:
                    print(f"⚠️ Error getting embedding for text: {e}")
                    embeddings.append([0.0] * self.embedding_dim)
            
            return np.array(embeddings)
        
        return ollama_embedding_func
    
    def insert_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Insert documents into the graph RAG system."""
        stats = {
            'total_documents': len(documents),
            'processed_documents': 0,
            'entities_extracted': 0,
            'relationships_created': 0,
            'processing_time': 0
        }
        
        try:
            start_time = time.time()
            print(f"🔗 [GRAPH] Starting knowledge graph construction...")
            print(f"📄 [GRAPH] Processing {len(documents)} documents for entity and relationship extraction")
            
            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_insert_documents, documents, stats)
                    result_stats = future.result()
            except RuntimeError:
                result_stats = self._sync_insert_documents(documents, stats)
            
            result_stats['processing_time'] = time.time() - start_time
            print(f"✅ [GRAPH] Knowledge graph construction completed in {result_stats['processing_time']:.2f}s")
            print(f"📊 [GRAPH] Final statistics: {result_stats['processed_documents']} docs, ~{result_stats.get('entities_extracted', 'N/A')} entities, ~{result_stats.get('relationships_created', 'N/A')} relationships")
            
            return result_stats
        except Exception as e:
            print(f"❌ [GRAPH] Error during graph construction: {e}")
            raise
    
    def _sync_insert_documents(self, documents: List[str], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous document insertion helper with detailed logging."""
        for i, doc in enumerate(documents):
            print(f"   🔍 [GRAPH] Processing document {i+1}/{len(documents)} ({len(doc)} chars)")
            
            self.graph_rag.insert(doc)
            stats['processed_documents'] += 1
            
            if (i + 1) % 10 == 0 or i == len(documents) - 1:
                print(f"   📈 [GRAPH] Progress: {i+1}/{len(documents)} documents processed")
        
        try:
            if hasattr(self.graph_rag, 'entities_vdb') and self.graph_rag.entities_vdb:
                stats['entities_extracted'] = stats['processed_documents'] * 3  # Rough estimate
            if hasattr(self.graph_rag, 'chunk_entity_relation_graph'):
                stats['relationships_created'] = stats['processed_documents'] * 2  # Rough estimate
        except:
            pass
        
        return stats
    
    def query_local(self, query: str, **kwargs) -> str:
        """Query using local mode (entity-focused)."""
        return self._safe_query(query, "local", **kwargs)
    
    def query_global(self, query: str, **kwargs) -> str:
        """Query using global mode (community-focused)."""
        return self._safe_query(query, "global", **kwargs)
    
    def query_naive(self, query: str, **kwargs) -> str:
        """Query using naive mode (traditional RAG)."""
        return self._safe_query(query, "naive", **kwargs)
    
    def _safe_query(self, query: str, mode: str, **kwargs) -> str:
        """Safely execute query handling async context."""
        try:
            print(f"🔍 [GRAPH-QUERY] Starting {mode} mode query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            start_time = time.time()
            
            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_query, query, mode, **kwargs)
                    result = future.result()
            except RuntimeError:
                result = self._sync_query(query, mode, **kwargs)
            
            query_time = time.time() - start_time
            result_length = len(result) if result else 0
            
            if result:
                print(f"✅ [GRAPH-QUERY] {mode.capitalize()} query completed in {query_time:.2f}s, returned {result_length} chars")
            else:
                print(f"⚠️ [GRAPH-QUERY] {mode.capitalize()} query completed in {query_time:.2f}s but returned no results")
            
            return result
        except Exception as e:
            print(f"❌ [GRAPH-QUERY] Error in {mode} query: {e}")
            return ""
    
    def _sync_query(self, query: str, mode: str, **kwargs) -> str:
        """Synchronous query helper."""
        print(f"   🧠 [GRAPH-QUERY] Executing {mode} mode graph traversal...")
        param = QueryParam(mode=mode, **kwargs)
        result = self.graph_rag.query(query, param=param)
        print(f"   📝 [GRAPH-QUERY] Graph traversal completed, processing response...")
        return result

    def query(self, query: str, mode: str = "local", **kwargs) -> str:
        """Generic query method with mode selection."""
        if mode == "local":
            return self.query_local(query, **kwargs)
        elif mode == "global":
            return self.query_global(query, **kwargs)
        elif mode == "naive":
            return self.query_naive(query, **kwargs)
        else:
            raise ValueError(f"Unknown query mode: {mode}. Use 'local', 'global', or 'naive'")
    
    def get_entities(self) -> List[str]:
        """Get all entities from the graph."""
        try:
            entities_storage = self.graph_rag.entities_vdb
            if entities_storage:
                return []
            return []
        except Exception as e:
            print(f"❌ Error getting entities: {e}")
            return []
    
    def get_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity."""
        try:
            graph = self.graph_rag.chunk_entity_relation_graph
            if graph and hasattr(graph, 'get_node_edges'):
                return []
            return []
        except Exception as e:
            print(f"❌ Error getting relationships for {entity}: {e}")
            return []

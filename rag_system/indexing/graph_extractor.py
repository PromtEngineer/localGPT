from typing import List, Dict, Any
import json
from rag_system.utils.ollama_client import OllamaClient

class GraphExtractor:
    """
    Extracts entities and relationships from text chunks using a live Ollama model.
    """
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model
        print(f"Initialized GraphExtractor with Ollama model '{self.llm_model}'.")

    def extract(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        all_entities = {}
        all_relationships = set()

        print(f"Extracting graph from {len(chunks)} chunks with Ollama...")
        for i, chunk in enumerate(chunks):
            # Step 1: Extract Entities
            entity_prompt = f"""
            From the following text, extract key entities (people, companies, locations).
            Return the answer as a JSON object with a single key 'entities', which is a list of strings.
            Each entity should be a short, specific name, not a long string of text.

            Text: "{chunk['text']}"
            """
            
            entity_response = self.llm_client.generate_completion(
                self.llm_model, 
                entity_prompt,
                format="json" 
            )
            
            entity_response_text = entity_response.get('response', '{}')

            try:
                entity_data = json.loads(entity_response_text)
                entities = entity_data.get('entities', [])
                
                if not entities:
                    continue

                # Clean up entities
                cleaned_entities = []
                for entity in entities:
                    if len(entity) < 50 and not any(c in entity for c in "[]{}()"):
                        cleaned_entities.append(entity)

                if not cleaned_entities:
                    continue

                # Step 2: Extract Relationships
                relationship_prompt = f"""
                Given the following entities: {cleaned_entities}
                And the following text: "{chunk['text']}"
                Extract the relationships between the entities.
                Return the answer as a JSON object with a single key 'relationships', which is a list of objects, each with 'source', 'target', and 'label'.
                """

                relationship_response = self.llm_client.generate_completion(
                    self.llm_model,
                    relationship_prompt,
                    format="json"
                )

                relationship_response_text = relationship_response.get('response', '{}')
                relationship_data = json.loads(relationship_response_text)

                for entity_name in cleaned_entities:
                    all_entities[entity_name] = {"id": entity_name, "type": "Unknown"} # Placeholder type

                for rel in relationship_data.get("relationships", []):
                    if 'source' in rel and 'target' in rel and 'label' in rel:
                        all_relationships.add(
                            (rel['source'], rel['target'], rel['label'])
                        )

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from LLM for chunk {i+1}.")
                continue
        
        return {
            "entities": list(all_entities.values()),
            "relationships": [{"source": s, "target": t, "label": l} for s, t, l in all_relationships]
        }

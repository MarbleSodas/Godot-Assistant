import ast
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def generate_skeleton(file_path: str) -> str:
    """
    Parses a Python file and returns ONLY class/func signatures + docstrings.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return f"# Syntax Error parsing {os.path.basename(file_path)}"
        
        skeleton = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Get the signature
                name = node.name
                doc = ast.get_docstring(node)
                
                # Format docstring (truncate if too long)
                if doc:
                    doc = doc.strip()
                    if len(doc) > 200:
                        doc = doc[:197] + "..."
                    doc_str = f'  """{doc}"""'
                else:
                    doc_str = '  # No docstring'
                
                # Handle function args
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    skeleton.append(f"def {name}({', '.join(args)}):")
                    skeleton.append(doc_str)
                
                # Handle class
                elif isinstance(node, ast.ClassDef):
                    bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    base_str = f"({', '.join(bases)})" if bases else ""
                    skeleton.append(f"class {name}{base_str}:")
                    skeleton.append(doc_str)
                    
                    # Get methods
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef):
                            sub_args = [a.arg for a in sub.args.args]
                            
                            # Check for decorators (e.g. @property, @staticmethod)
                            decorators = []
                            for dec in sub.decorator_list:
                                if isinstance(dec, ast.Name):
                                    decorators.append(f"@{dec.id}")
                            
                            for dec in decorators:
                                skeleton.append(f"    {dec}")
                                
                            skeleton.append(f"    def {sub.name}({', '.join(sub_args)}):")
                            sub_doc = ast.get_docstring(sub)
                            if sub_doc:
                                sub_doc = sub_doc.strip().split('\n')[0] # First line only for methods
                                if len(sub_doc) > 100:
                                    sub_doc = sub_doc[:97] + "..."
                                skeleton.append(f"      # {sub_doc}")

    except Exception as e:
        return f"# Error processing {os.path.basename(file_path)}: {e}"
    
    return "\n".join(skeleton)

class ContextEngine:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.map_cache: Dict[str, str] = {}  # Tier 1 Cache
        self._map_generated = False

    def build_map(self) -> str:
        """Tier 1: Generates the high-level skeleton of the project."""
        # Return cached map if available
        if self._map_generated and self.map_cache:
            return self._format_map()

        skeleton_summary = []
        try:
            for root, _, files in os.walk(self.project_path):
                # Skip venv, .git, __pycache__, etc.
                if any(skip in root for skip in ['.git', '__pycache__', 'venv', 'node_modules', '.godoty']):
                    continue
                    
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        try:
                            rel_path = os.path.relpath(full_path, self.project_path)
                            
                            # Generate skeleton
                            skel = generate_skeleton(full_path)
                            if skel.strip(): # Only add if not empty
                                self.map_cache[rel_path] = skel
                        except Exception as e:
                            logger.warning(f"Failed to process {file}: {e}")
            
            self._map_generated = True
            return self._format_map()
            
        except Exception as e:
            logger.error(f"Error building context map: {e}")
            return "# Error building context map"

    def _format_map(self) -> str:
        """Format the cached map into a string."""
        summary = []
        for path, skel in sorted(self.map_cache.items()):
            summary.append(f"### FILE: {path}\n{skel}")
        return "\n\n".join(summary)

    def retrieve_relevant_context(self, query: str) -> str:
        """
        Tier 3: Uses RAG to find specific implementation details 
        based on the user's query.
        """
        # Placeholder for RAG implementation
        # In a real implementation, this would query a vector database
        return "" 

    def get_context_for_prompt(self, user_query: str) -> str:
        """
        Combines Tier 1 (Always on) + Tier 3 (On demand)
        """
        project_map = self.build_map()
        specifics = self.retrieve_relevant_context(user_query)
        
        context_parts = [
            "PROJECT MAP (Structure):",
            project_map
        ]
        
        if specifics:
            context_parts.extend([
                "\nRELEVANT CODE (Implementation):",
                specifics
            ])
            
        return "\n".join(context_parts)

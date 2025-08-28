"""
OmicVerse Smart Agent using Pantheon Framework

This module provides a smart agent that can understand natural language requests
and automatically execute appropriate OmicVerse functions.

Usage:
    import omicverse as ov
    result = ov.Agent("quality control with nUMI>500, mito<0.2", adata)
"""

import sys
import os
import asyncio
import json
import re
import inspect
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Add pantheon path if not already in path
pantheon_path = '/Users/fernandozeng/Desktop/analysis/pantheon/pantheon-agents'
if pantheon_path not in sys.path:
    sys.path.append(pantheon_path)

from pantheon.agent import Agent as PantheonAgent
from pantheon.toolsets.python import PythonInterpreterToolSet

# Import registry system
from .registry import _global_registry


class OmicVerseAgent:
    """
    Intelligent agent for OmicVerse function discovery and execution.
    
    This agent uses the Pantheon framework to understand natural language
    requests and automatically execute appropriate OmicVerse functions.
    
    Usage:
        agent = ov.Agent(model="gpt-4o-mini", api_key="your-api-key")
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize the OmicVerse Smart Agent.
        
        Parameters
        ----------
        model : str
            LLM model to use for reasoning (default: "gpt-4o-mini")
        api_key : str, optional
            OpenAI API key. If not provided, will use environment variable
        """
        self.model = model
        self.api_key = api_key
        self.agent = None
        
        # Set API key if provided
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key
        
        self._setup_agent()
    
    def _get_available_functions_info(self) -> str:
        """Get formatted information about all available functions."""
        functions_info = []
        
        # Get all unique functions from registry
        processed_functions = set()
        for entry in _global_registry._registry.values():
            full_name = entry['full_name']
            if full_name in processed_functions:
                continue
            processed_functions.add(full_name)
            
            # Format function information
            info = {
                'name': entry['short_name'],
                'full_name': entry['full_name'],
                'description': entry['description'],
                'aliases': entry['aliases'],
                'category': entry['category'],
                'signature': entry['signature'],
                'examples': entry['examples']
            }
            functions_info.append(info)
        
        return json.dumps(functions_info, indent=2, ensure_ascii=False)
    
    def _setup_agent(self):
        """Setup the pantheon agent with dynamic instructions."""
        
        # Get current function information dynamically
        functions_info = self._get_available_functions_info()
        
        instructions = """
You are an intelligent OmicVerse assistant that can automatically discover and execute functions based on natural language requests.

## Available OmicVerse Functions

Here are all the currently registered functions in OmicVerse:

""" + functions_info + """

## Your Task

When given a natural language request and an adata object, you should:

1. **Analyze the request** to understand what the user wants to accomplish
2. **Find the most appropriate function** from the available functions above
3. **Extract parameters** from the user's request (e.g., "nUMI>500" means min_genes=500)
4. **Generate and execute Python code** using the appropriate OmicVerse function
5. **Return the modified adata object**

## Parameter Extraction Rules

Extract parameters dynamically based on patterns in the user request:

- For qc function: Create tresh dict with 'mito_perc', 'nUMIs', 'detected_genes'
  - "nUMI>X", "umi>X" → tresh={'nUMIs': X, 'detected_genes': 250, 'mito_perc': 0.15}
  - "mito<X", "mitochondrial<X" → include in tresh dict as 'mito_perc': X
  - "genes>X" → include in tresh dict as 'detected_genes': X
  - Always provide complete tresh dict with all three keys
- "resolution=X" → resolution=X
- "n_pcs=X", "pca=X" → n_pcs=X
- "max_value=X" → max_value=X
- Mode indicators: "seurat", "mads", "pearson" → mode="seurat"
- Boolean indicators: "no doublets", "skip doublets" → doublets=False

## Code Execution Rules

1. **Always import omicverse as ov** at the start
2. **Use the exact function signature** from the available functions
3. **Handle the adata variable** - it will be provided in the context
4. **Update adata in place** when possible
5. **Print success messages** and basic info about the result

## Example Workflow

User request: "quality control with nUMI>500, mito<0.2"

1. Find function: Look for functions with aliases containing "qc", "quality", or "质控"
2. Get function details: Check that qc requires tresh dict with 'mito_perc', 'nUMIs', 'detected_genes'
3. Extract parameters: nUMI>500 → tresh['nUMIs']=500, mito<0.2 → tresh['mito_perc']=0.2
4. Generate code:
   ```python
   import omicverse as ov
   # Execute quality control with complete tresh dict
   adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})
   print("QC completed. Dataset shape: " + str(adata.shape[0]) + " cells × " + str(adata.shape[1]) + " genes")
   ```

## Important Notes

- Always work with the provided `adata` variable
- Use the function signatures exactly as shown in the available functions
- Provide helpful feedback about what was executed
- Handle errors gracefully and suggest alternatives if needed
"""
        
        # Setup Python interpreter
        python_toolset = PythonInterpreterToolSet("python")
        
        # Create the pantheon agent
        self.agent = PantheonAgent(
            "omicverse_agent",
            instructions,
            model=self.model,
        )
        
        # Add toolsets
        self.agent.toolset(python_toolset)
        
        # Add custom tools for function discovery
        self.agent.tool(self._search_functions)
        self.agent.tool(self._get_function_details)
    
    def _search_functions(self, query: str) -> str:
        """
        Search for functions in the OmicVerse registry.
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        str
            JSON formatted search results
        """
        try:
            results = _global_registry.find(query)
            
            if not results:
                return json.dumps({"error": f"No functions found for query: '{query}'"})
            
            # Format results for the agent
            formatted_results = []
            for entry in results:
                formatted_results.append({
                    'name': entry['short_name'],
                    'full_name': entry['full_name'],
                    'description': entry['description'],
                    'signature': entry['signature'],
                    'aliases': entry['aliases'],
                    'examples': entry['examples'],
                    'category': entry['category']
                })
            
            return json.dumps({
                "found": len(formatted_results),
                "functions": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error searching functions: {str(e)}"})
    
    def _get_function_details(self, function_name: str) -> str:
        """
        Get detailed information about a specific function.
        
        Parameters
        ----------
        function_name : str
            Function name or alias
            
        Returns
        -------
        str
            JSON formatted function details
        """
        try:
            results = _global_registry.find(function_name)
            
            if not results:
                return json.dumps({"error": f"Function '{function_name}' not found"})
            
            entry = results[0]  # Get first match
            
            return json.dumps({
                'name': entry['short_name'],
                'full_name': entry['full_name'],
                'description': entry['description'],
                'signature': entry['signature'],
                'parameters': entry.get('parameters', []),
                'aliases': entry['aliases'],
                'examples': entry['examples'],
                'category': entry['category'],
                'docstring': entry['docstring'],
                'help': f"Function: {entry['full_name']}\nSignature: {entry['signature']}\n\nDescription:\n{entry['description']}\n\nDocstring:\n{entry['docstring']}\n\nExamples:\n" + "\n".join(entry['examples'])
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error getting function details: {str(e)}"})
    
    async def run_async(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request and execute the generated code locally.
        
        Parameters
        ----------
        request : str
            Natural language description of what to do
        adata : Any
            AnnData object to process
            
        Returns
        -------
        Any
            Processed adata object
        """
        
        # Ask agent to generate the appropriate function call code
        code_generation_request = f'''
Please analyze this OmicVerse request: "{request}"

Your task:
1. Use the _search_functions tool to find the most appropriate OmicVerse function
2. Use the _get_function_details tool to get the complete function signature and docstring
3. Based on the function details, extract parameters from the request text
4. Generate executable Python code that calls the correct OmicVerse function with proper parameters

Dataset info:
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
- Request: {request}

CRITICAL INSTRUCTIONS:
1. ALWAYS call _search_functions first to find the right function
2. ALWAYS call _get_function_details to get complete help info before generating code
3. Read the 'help' field carefully to understand all parameters and their defaults
4. Generate code that matches the actual function signature
5. Return ONLY executable Python code, no explanations

For the qc function specifically:
- The tresh parameter needs a dict with 'mito_perc', 'nUMIs', 'detected_genes' keys
- Default is: tresh={{'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}}
- Extract values from user request and update the dict accordingly

Example workflow:
1. _search_functions("quality control") → finds ov.pp.qc
2. _get_function_details("qc") → read help to see all parameters
3. Parse request: "nUMI>500" means tresh['nUMIs']=500
4. Generate: ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
'''
        
        # Get the code from the agent
        print(f"\n🤔 Agent analyzing request: '{request}'...")
        response = await self.agent.run(code_generation_request)
        
        # Extract code from the response
        # Check if response has content attribute (Pantheon agent response)
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Display agent's response
        print(f"\n💭 Agent response:")
        print("-" * 50)
        # Only show the actual content, not the full object
        if hasattr(response, 'content'):
            print(f"{response.content}")
        else:
            print(response_text)
        print("-" * 50)
        
        # Find code blocks in the response
        code_blocks = []
        import re
        
        # Look for ```python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
        code_blocks.extend(python_blocks)
        
        # Look for ``` code blocks without language specification
        generic_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
        code_blocks.extend(generic_blocks)
        
        if not code_blocks:
            # If no code blocks found, check if response_text is already clean code
            # Extract all Python code lines including variable definitions
            clean_lines = []
            for line in response_text.split('\n'):
                line_stripped = line.strip()
                # Skip empty lines and non-code lines
                if not line_stripped:
                    continue
                # Include lines that are Python code
                if any([
                    line_stripped.startswith('import '),
                    line_stripped.startswith('from '),
                    line_stripped.startswith('adata'),
                    line_stripped.startswith('tresh'),
                    '=' in line_stripped,  # Variable assignments
                    'ov.' in line_stripped,
                    line_stripped.startswith('try:'),
                    line_stripped.startswith('except'),
                ]):
                    clean_lines.append(line_stripped)
            
            if clean_lines:
                # Ensure import is at the top
                has_import = any('import omicverse' in line for line in clean_lines)
                if not has_import:
                    code = "import omicverse as ov\n" + '\n'.join(clean_lines)
                else:
                    code = '\n'.join(clean_lines)
                
                # Filter out lines that shouldn't be in final code (indentation issues)
                # Only keep the essential lines
                essential_lines = []
                for line in code.split('\n'):
                    # Skip try/except blocks that might have indentation issues
                    if line.strip().startswith('try:') or line.strip().startswith('except'):
                        continue
                    if line.strip().startswith('pc_count'):
                        continue
                    essential_lines.append(line)
                    
                # Keep only the core function calls
                if essential_lines:
                    code = '\n'.join(essential_lines)
            else:
                raise ValueError(f"❌ Could not extract executable code from agent response")
        else:
            # Use the first code block found
            code = code_blocks[0].strip()
        
        print(f"\n🧬 Generated code to execute:")
        print("=" * 50)
        print(f"{code}")
        print("=" * 50)
        
        # Execute the code locally
        print(f"\n⚡ Executing code locally...")
        try:
            # Create execution context with adata
            local_vars = {'adata': adata}
            exec(code, globals(), local_vars)
            
            # Return the modified adata
            result_adata = local_vars.get('adata', adata)
            print(f"✅ Code executed successfully!")
            print(f"📊 Result shape: {result_adata.shape[0]} cells × {result_adata.shape[1]} genes")
            
            return result_adata
            
        except Exception as e:
            print(f"❌ Error executing generated code: {e}")
            print(f"Code that failed: {code}")
            return adata
    
    def run(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request with the provided adata (main method).
        
        Parameters
        ----------
        request : str
            Natural language description of what to do
        adata : Any
            AnnData object to process
            
        Returns
        -------
        Any
            Processed adata object (modified)
            
        Examples
        --------
        >>> agent = ov.Agent(model="gpt-4o-mini")
        >>> result = agent.run("quality control with nUMI>500, mito<0.2", adata)
        """
        return asyncio.run(self.run_async(request, adata))


def Agent(model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> OmicVerseAgent:
    """
    Create an OmicVerse Smart Agent instance.
    
    This function creates and returns a smart agent that can execute OmicVerse functions
    based on natural language descriptions.
    
    Parameters
    ----------
    model : str, optional
        LLM model to use (default: "gpt-4o-mini")
    api_key : str, optional
        OpenAI API key. If not provided, will use environment variable
        
    Returns
    -------
    OmicVerseAgent
        Configured agent instance ready for use
        
    Examples
    --------
    >>> import omicverse as ov
    >>> import scanpy as sc
    >>> 
    >>> # Create agent instance
    >>> agent = ov.Agent(model="gpt-4o-mini", api_key="your-key")
    >>> 
    >>> # Load data
    >>> adata = sc.datasets.pbmc3k()
    >>> 
    >>> # Use agent for quality control
    >>> adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    >>> 
    >>> # Use agent for preprocessing  
    >>> adata = agent.run("preprocess with 2000 highly variable genes", adata)
    >>> 
    >>> # Use agent for clustering
    >>> adata = agent.run("leiden clustering resolution=1.0", adata)
    """
    return OmicVerseAgent(model=model, api_key=api_key)


# Export the main function
__all__ = ['Agent', 'OmicVerseAgent']
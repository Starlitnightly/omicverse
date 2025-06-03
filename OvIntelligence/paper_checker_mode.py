# --- START OF FILE paper_checker_mode.py ---
import streamlit as st
import ast
import os
from google import genai
from google.genai import types
from token_counter import count_tokens_decorator
from logger_setup import logger
from rag_system import RAGSystem, PackageConfig

# NOTE: For production, set your GEMINI_API_KEY in the environment.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_DEFAULT_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def clean_markdown(text: str) -> str:
    """Remove markdown code fences from the text."""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def extract_candidate_text(text: str) -> str:
    """Extract the substring between the first '[' and the last ']'."""
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx + 1]
    return text


def generate_stage1_prompt(paper_content: str) -> str:
    """
    Create the prompt for Stage 1 (text mode) for plot identification.
    (Tool intros have been shortened here for brevity.)
    """
    return f"""
You are an expert in single-cell bioinformatics.
Given the following text from a scientific paper, identify any single-cell plots or figures.
For each identified plot, provide:
- plot_name: a short descriptive name for the plot.
- analysis_name: the type of single-cell analysis or data involved.
- selected_package: choose the best-suited Python package from: CellRank, scvi-tools, Spateo, Squidpy, OmicVerse.
Here's a breakdown of each of these Python-based single-cell analysis tools, outlining their primary functions and key features:

1. CellRank
Purpose: CellRank is designed to infer cellular dynamics and fate decisions in single-cell data. It models cell state transitions as a Markov process, helping to understand developmental trajectories, reprogramming, and responses to perturbations.
Key Features:
Trajectory Inference: Predicts future cell states and lineage commitments.
Driver Gene Identification: Uncovers genes driving transitions between cell states.
RNA Velocity Integration: Combines RNA velocity (which infers the direction of cellular changes based on spliced and unspliced mRNA) with other data modalities.
Different Kernels: allows various kernels, including a 'VelocityKernel', 'ConnectivityKernel', and a 'CombinedKernel'.

2. scvi-tools (Single-Cell Variational Inference Tools)
Purpose: scvi-tools leverages probabilistic models, particularly variational autoencoders (VAEs), for deep learning-based analysis of single-cell data. It excels at handling technical noise and complex datasets.
Key Features:
Data Integration: Integrates data from multiple batches, experiments, or modalities.
Batch Effect Correction: Corrects for unwanted technical variations between datasets.
Dimensionality Reduction: Latent space representation learning.
Differential Expression: Probabilistic differential expression analysis.
Multi-omics Integration: Designed to integrate various omics data types (e.g., RNA-seq, ATAC-seq).
Uncertainty Quantification: Models uncertainty in the analysis, providing more robust results.

3. Spateo
Purpose: Spateo is specifically designed for analyzing spatial transcriptomics data. Spatial transcriptomics captures both gene expression and the physical location of cells within a tissue.
Key Features:
Preprocessing: Designed for data generated with various spatial platforms, like 10x Genomics Xenium, 10x Genomics Visium, and Nanostring CosMx.
Analysis: Spateo can be used for a wide range of applications, including characterizing tissue heterogeneity, studying cell-cell communication, and understanding how cellular processes vary across space.
Dynamical Modeling: Models the dynamics of gene expression in tissues.
Morphological Data: Takes advantage of the morphological data.

4. Squidpy
Purpose: Squidpy is another library focused on the analysis and visualization of spatial omics data. It integrates well with Scanpy and AnnData (Annotated Data).
Key Features:
Spatial Data Analysis: Tools for analyzing spatial relationships between cells, including spatial autocorrelation, co-occurrence, and neighbor analysis.
Image Processing: Integrates with image processing tools for handling imaging-based spatial data.
Cell-Cell Interaction: Analysis of ligand-receptor interactions and communication patterns.
Graph-Based Analysis: Leverages graph representations of spatial data for analysis and visualization.
Interactive Visualization: Provides interactive tools for exploring spatial data.

5. OmicVerse
Purpose: Designed for multi-omics analysis.
Key Features:
Multi-Omics Data. Tools to work on a wide range of multi-omics data, inclduing single-cell RNA-seq data.
Integration with Scanpy. Compatible with the Scanpy package.
Metabolic Functionality. Tools to analyze metabolomics.

Return a valid Python literal (a list of dictionaries) like:
[
  {{
    "plot_name": "...",
    "analysis_name": "...",
    "selected_package": "..."
  }},
  ...
]
Paper content:
{paper_content}
"""


def generate_stage1_pdf_prompt() -> str:
    """
    Create the prompt for Stage 1 (PDF mode) for plot identification.
    (Tool intros have been shortened here for brevity.)
    """
    return f"""
You are an expert in single-cell bioinformatics.
Given the following text from a scientific paper, identify any single-cell plots or figures.
For each identified plot, provide:
- plot_name: a short descriptive name for the plot.
- analysis_name: the type of single-cell analysis or data involved.
- selected_package: choose the best-suited Python package from: CellRank, scvi-tools, Spateo, Squidpy, OmicVerse.
Here's a breakdown of each of these Python-based single-cell analysis tools, outlining their primary functions and key features:

1. CellRank
Purpose: CellRank is designed to infer cellular dynamics and fate decisions in single-cell data. It models cell state transitions as a Markov process, helping to understand developmental trajectories, reprogramming, and responses to perturbations.
Key Features:
Trajectory Inference: Predicts future cell states and lineage commitments.
Driver Gene Identification: Uncovers genes driving transitions between cell states.
RNA Velocity Integration: Combines RNA velocity (which infers the direction of cellular changes based on spliced and unspliced mRNA) with other data modalities.
Different Kernels: allows various kernels, including a 'VelocityKernel', 'ConnectivityKernel', and a 'CombinedKernel'.

2. scvi-tools (Single-Cell Variational Inference Tools)
Purpose: scvi-tools leverages probabilistic models, particularly variational autoencoders (VAEs), for deep learning-based analysis of single-cell data. It excels at handling technical noise and complex datasets.
Key Features:
Data Integration: Integrates data from multiple batches, experiments, or modalities.
Batch Effect Correction: Corrects for unwanted technical variations between datasets.
Dimensionality Reduction: Latent space representation learning.
Differential Expression: Probabilistic differential expression analysis.
Multi-omics Integration: Designed to integrate various omics data types (e.g., RNA-seq, ATAC-seq).
Uncertainty Quantification: Models uncertainty in the analysis, providing more robust results.

3. Spateo
Purpose: Spateo is specifically designed for analyzing spatial transcriptomics data. Spatial transcriptomics captures both gene expression and the physical location of cells within a tissue.
Key Features:
Preprocessing: Designed for data generated with various spatial platforms, like 10x Genomics Xenium, 10x Genomics Visium, and Nanostring CosMx.
Analysis: Spateo can be used for a wide range of applications, including characterizing tissue heterogeneity, studying cell-cell communication, and understanding how cellular processes vary across space.
Dynamical Modeling: Models the dynamics of gene expression in tissues.
Morphological Data: Takes advantage of the morphological data.

4. Squidpy
Purpose: Squidpy is another library focused on the analysis and visualization of spatial omics data. It integrates well with Scanpy and AnnData (Annotated Data).
Key Features:
Spatial Data Analysis: Tools for analyzing spatial relationships between cells, including spatial autocorrelation, co-occurrence, and neighbor analysis.
Image Processing: Integrates with image processing tools for handling imaging-based spatial data.
Cell-Cell Interaction: Analysis of ligand-receptor interactions and communication patterns.
Graph-Based Analysis: Leverages graph representations of spatial data for analysis and visualization.
Interactive Visualization: Provides interactive tools for exploring spatial data.

5. OmicVerse
Purpose: Designed for multi-omics analysis.
Key Features:
Multi-Omics Data. Tools to work on a wide range of multi-omics data, inclduing single-cell RNA-seq data.
Integration with Scanpy. Compatible with the Scanpy package.
Metabolic Functionality. Tools to analyze metabolomics.

Return a valid Python literal (a list of dictionaries) like:
[
  {{
    "plot_name": "...",
    "analysis_name": "...",
    "selected_package": "..."
  }},
  ...
]
"""


def process_llm_response(response) -> list:
    """Process the Gemini model response and return a list of plot info."""
    raw_text = response.text.strip() if hasattr(response, "text") and response.text else ""
    if not raw_text:
        logger.warning("Gemini model returned an empty response.")
        return []
    cleaned_text = clean_markdown(raw_text)
    parsed_text = extract_candidate_text(cleaned_text)
    try:
        results = ast.literal_eval(parsed_text)
        if isinstance(results, list):
            return results
        else:
            logger.error("Parsed result is not a list.")
            return []
    except Exception as e:
        logger.error("Failed to parse candidate text.", exc_info=True)
        return []


@count_tokens_decorator
def paperchecker_stage1_llm(paper_content: str = "", pdf_bytes: bytes = None) -> list:
    """
    Stage 1: Identify single-cell plots using the Gemini model.
    Supports both plain text input and PDF file bytes.
    """
    if pdf_bytes is not None:
        try:
            prompt_text = generate_stage1_pdf_prompt()
            response = client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                    prompt_text
                ]
            )
            return process_llm_response(response)
        except Exception as e:
            logger.error(f"Error in paperchecker_stage1_llm with PDF input: {e}", exc_info=True)
            return []
    # Text mode
    if not paper_content.strip():
        return []
    try:
        prompt_text = generate_stage1_prompt(paper_content)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_text
        )
        if not response.candidates:
            logger.warning("No candidates returned from the Gemini model.")
            return []
        return process_llm_response(response)
    except Exception as e:
        logger.error(f"Error in paperchecker_stage1_llm: {e}", exc_info=True)
        return []


def get_rag_system_instance() -> RAGSystem:
    """Return a cached RAGSystem instance, creating it if needed.

    The original implementation used hard-coded absolute paths which often do
    not exist on the running machine.  This version optionally reads the base
    path from the ``OV_DATA_ROOT`` environment variable and silently skips any
    packages whose directories are missing.  This prevents immediate failures
    and allows Paper Checker Mode to run even with partial data.
    """
    if 'rag_system' in st.session_state:
        return st.session_state['rag_system']

    data_root = os.environ.get("OV_DATA_ROOT", "/Users/kq_m3m/PycharmProjects/SCMaster")
    package_defs = [
        ("CellRank", "cellrank_notebooks"),
        ("Scanpy", "scanpy-tutorials"),
        ("scvi-tools", "scvi-tutorials"),
        ("Spateo", "spateo-tutorials"),
        ("Squidpy", "squidpy_notebooks"),
        ("OmicVerse", "ov_tut"),
    ]

    configs = []
    for name, subdir in package_defs:
        conv_dir = os.path.join(data_root, "6O_json_files", subdir)
        annot_dir = os.path.join(data_root, "annotated_scripts", subdir)
        if os.path.isdir(conv_dir) and os.path.isdir(annot_dir):
            configs.append(
                PackageConfig(
                    name=name,
                    converted_jsons_directory=conv_dir,
                    annotated_scripts_directory=annot_dir,
                    file_selection_model="gemini-2.0-flash-lite-preview-02-05",
                    query_processing_model="gemini-2.0-flash-lite-preview-02-05",
                )
            )
        else:
            logger.warning(
                f"Skipping package '{name}' due to missing directories: {conv_dir} or {annot_dir}"
            )

    if not configs:
        logger.error("No valid package directories found for RAGSystem. Paper Checker mode will be limited.")

    rag_system_instance = RAGSystem(configs)
    st.session_state['rag_system'] = rag_system_instance
    return rag_system_instance


@count_tokens_decorator
def paperchecker_stage2_llm(stage1_results: list, user_prompt: str) -> dict:
    """
    Stage 2: For each plot identified in Stage 1, use the integrated RAG system to:
      - Retrieve relevant code examples based on the query.
      - Generate and refine a code snippet using the selected package.
    Returns a dictionary with a list of code instructions.
    Each code instruction now includes both an example Python code snippet and a code explanation.
    """
    if not stage1_results:
        return {
            "code_instructions": [{
                "plot_name": "N/A",
                "analysis_name": "N/A",
                "selected_package": "OmicVerse",
                "code_snippet": "No Stage1 results were provided.",
                "code_explanation": "No code explanation available."
            }]
        }
    code_instructions = []
    rag_system_instance = get_rag_system_instance()
    for plot_info in stage1_results:
        plot_name = plot_info.get("plot_name", "N/A")
        analysis_name = plot_info.get("analysis_name", "N/A")
        selected_package = plot_info.get("selected_package", "N/A")
        # Construct a query if no user prompt is provided.
        query = user_prompt.strip() if user_prompt.strip() else (
            f"Generate a code snippet for a plot named '{plot_name}' with analysis '{analysis_name}'."
        )
        try:
            refined_code = rag_system_instance.generate_and_refine_code_sample(plot_info, query)
        except Exception as e:
            logger.error(f"Error generating code for plot '{plot_name}': {e}", exc_info=True)
            refined_code = f"Error during code generation: {e}"
        # If the refined_code is a dict, extract both parts; otherwise, set a fallback.
        if isinstance(refined_code, dict):
            snippet = refined_code.get("code_snippet", "")
            explanation = refined_code.get("code_explanation", "No explanation provided.")
        else:
            snippet = refined_code
            explanation = "No explanation provided."
        code_instructions.append({
            "plot_name": plot_name,
            "analysis_name": analysis_name,
            "selected_package": selected_package,
            "code_snippet": snippet,
            "code_explanation": explanation
        })
    return {"code_instructions": code_instructions}


def paper_checker_interface():
    """
    Streamlit interface for the two-stage Paper Checker Mode.
    Supports both text and PDF input for Stage 1.
    """
    st.markdown("## Paper Checker Mode")
    st.info(
        "This mode uses a two-stage LLM approach: first, identify single-cell plots; then, generate refined code instructions using a RAG + Gemini pipeline.")

    # --- Stage 1: Plot Identification ---
    st.subheader("Stage 1: Identify Single-Cell Plots")
    input_mode = st.radio("Select input mode", options=["Text", "PDF"], index=0)
    if input_mode == "Text":
        paper_text = st.text_area("Paste the paper content:", value=st.session_state.get('paper_content', ""),
                                  height=200)
        pdf_file = None
    else:
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        paper_text = ""

    if st.button("Run Stage 1 (Identify Plots)"):
        if input_mode == "Text":
            if paper_text.strip():
                with st.spinner("Analyzing paper content..."):
                    stage1_results = paperchecker_stage1_llm(paper_content=paper_text)
                    st.session_state['paper_content'] = paper_text
                    st.session_state['paper_stage1_results'] = stage1_results
                    if stage1_results:
                        st.success("Found single-cell plot information!")
                    else:
                        st.warning("No relevant plots found or parsing failed.")
            else:
                st.error("Paper content is empty. Please provide valid text.")
        else:
            if pdf_file is not None:
                pdf_bytes = pdf_file.read()
                with st.spinner("Analyzing PDF content..."):
                    stage1_results = paperchecker_stage1_llm(pdf_bytes=pdf_bytes)
                    st.session_state['paper_stage1_results'] = stage1_results
                    if stage1_results:
                        st.success("Found single-cell plot information!")
                    else:
                        st.warning("No relevant plots found or parsing failed.")
            else:
                st.error("No PDF file uploaded. Please upload a PDF.")

    stage1_results = st.session_state.get('paper_stage1_results', [])
    if stage1_results:
        st.write("**Detected Single-Cell Plots:**")
        for idx, item in enumerate(stage1_results):
            st.markdown(f"**Plot {idx + 1}:**")
            st.write(f"- **Plot Name:** {item.get('plot_name', 'N/A')}")
            st.write(f"- **Analysis:** {item.get('analysis_name', 'N/A')}")
            st.write(f"- **Selected Package:** {item.get('selected_package', 'N/A')}")
            st.markdown("---")

    # --- Stage 2: Code Generation ---
    st.subheader("Stage 2: Generate Refined Code Snippets")
    user_prompt = st.text_input("Additional instructions or context:", "")
    if st.button("Run Stage 2 (Generate Code)"):
        if not stage1_results:
            st.error("No Stage 1 results found. Please run Stage 1 first.")
        else:
            with st.spinner("Generating code instructions..."):
                stage2_output = paperchecker_stage2_llm(stage1_results, user_prompt)
                code_instructions = stage2_output.get('code_instructions', [])
                st.markdown("### Code Instructions for Each Plot")
                if not code_instructions:
                    st.warning("No code instructions returned.")
                else:
                    for idx, snippet_info in enumerate(code_instructions):
                        expander_label = f"Plot {idx + 1}: {snippet_info.get('plot_name', 'N/A')}"
                        with st.expander(expander_label):
                            st.markdown(f"**Analysis:** {snippet_info.get('analysis_name', 'N/A')}")
                            st.markdown(f"**Selected Package:** {snippet_info.get('selected_package', 'N/A')}")

                            st.markdown("**Example Python Code:**")
                            st.code(snippet_info.get('code_snippet', ""), language='python')

                            st.markdown("**Code Explanation:**")
                            st.markdown(snippet_info.get('code_explanation', "No explanation provided."))
                            st.markdown("---")
# --- END OF FILE paper_checker_mode.py ---

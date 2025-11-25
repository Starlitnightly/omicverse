"""
Tests for code extraction robustness in OmicVerseAgent.

This module tests the agent's ability to extract Python code from various
LLM response formats, including:
- Fenced code blocks
- Inline code
- Async/await syntax
- Decorators
- Complex patterns
- Edge cases

Author: OmicVerse Development Team
Date: 2025-01-08
"""

import pytest
import ast
import textwrap
from omicverse.utils.smart_agent import OmicVerseAgent


class TestFencedCodeExtraction:
    """Test extraction from fenced code blocks (```python ... ```)"""

    def test_simple_fenced_block(self):
        """Test basic fenced code block"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        Here's the code:

        ```python
        import omicverse as ov
        adata = ov.pp.qc(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "import omicverse as ov" in code
        assert "ov.pp.qc(adata)" in code
        # Should be valid Python
        ast.parse(code)

    def test_fenced_without_language_specifier(self):
        """Test fenced block without 'python' keyword"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```
        import omicverse as ov
        adata = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "ov.pp.normalize" in code
        ast.parse(code)

    def test_multiple_fenced_blocks(self):
        """Test multiple fenced blocks (should extract first valid one)"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        First approach:
        ```python
        import omicverse as ov
        result1 = ov.pp.qc(adata)
        ```

        Alternative approach:
        ```python
        import omicverse as ov
        result2 = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        # Should extract at least one valid block
        assert ("result1" in code or "result2" in code or
                "ov.pp.qc" in code or "ov.pp.normalize" in code)
        ast.parse(code)

    def test_fenced_with_extra_whitespace(self):
        """Test fenced block with unusual whitespace"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python


        import omicverse as ov

        adata = ov.pp.qc(adata)


        ```
        """)

        code = agent._extract_python_code(response)
        assert "import omicverse" in code
        assert "ov.pp.qc" in code
        ast.parse(code)

    def test_fenced_with_comments(self):
        """Test fenced block with comments"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        # Preprocess the data
        import omicverse as ov

        # Quality control
        adata = ov.pp.qc(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "# Preprocess" in code
        assert "# Quality control" in code
        ast.parse(code)


class TestAsyncAwaitExtraction:
    """Test extraction of async/await syntax"""

    def test_async_function_definition(self):
        """Test async function extraction"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        async def process_data(adata):
            result = await ov.async_normalize(adata)
            return result

        adata = await process_data(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "async def process_data" in code
        assert "await ov.async_normalize" in code
        ast.parse(code)

    def test_async_context_manager(self):
        """Test async with statement"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        async with ov.async_processor() as processor:
            adata = await processor.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "async with" in code
        ast.parse(code)

    def test_async_comprehension(self):
        """Test async comprehensions"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        results = [await ov.process(x) async for x in data_stream]
        ```
        """)

        code = agent._extract_python_code(response)
        assert "async for" in code
        assert "await" in code
        ast.parse(code)


class TestDecoratorExtraction:
    """Test extraction of decorated functions"""

    def test_simple_decorator(self):
        """Test simple decorator"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        @ov.cache_result
        def process_data(adata):
            return ov.pp.normalize(adata)

        adata = process_data(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "@ov.cache_result" in code
        assert "def process_data" in code
        ast.parse(code)

    def test_decorator_with_arguments(self):
        """Test decorator with arguments"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        @ov.retry(max_attempts=3, delay=1.0)
        def unstable_process(adata):
            return ov.pp.qc(adata)

        adata = unstable_process(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "@ov.retry(max_attempts=3" in code
        ast.parse(code)

    def test_multiple_decorators(self):
        """Test multiple decorators on one function"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        @ov.cache
        @ov.validate_input
        @ov.log_execution
        def process_data(adata):
            return ov.pp.normalize(adata)

        adata = process_data(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "@ov.cache" in code
        assert "@ov.validate_input" in code
        assert "@ov.log_execution" in code
        ast.parse(code)

    def test_class_decorator(self):
        """Test class decorator"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        @ov.register_processor
        class CustomProcessor:
            def process(self, adata):
                return ov.pp.normalize(adata)

        processor = CustomProcessor()
        adata = processor.process(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "@ov.register_processor" in code
        assert "class CustomProcessor" in code
        ast.parse(code)


class TestComplexPatternsExtraction:
    """Test extraction of complex Python patterns"""

    def test_nested_function(self):
        """Test nested function definition"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        def outer_process(adata):
            def inner_normalize(data):
                return ov.pp.normalize(data)

            adata = inner_normalize(adata)
            return adata

        adata = outer_process(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "def outer_process" in code
        assert "def inner_normalize" in code
        ast.parse(code)

    def test_lambda_function(self):
        """Test lambda expressions"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        normalize = lambda x: ov.pp.normalize(x)
        adata = normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "lambda" in code
        ast.parse(code)

    def test_list_comprehension(self):
        """Test list comprehension"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        results = [ov.pp.normalize(x) for x in adata_list]
        ```
        """)

        code = agent._extract_python_code(response)
        assert "[ov.pp.normalize(x) for x in adata_list]" in code
        ast.parse(code)

    def test_dict_comprehension(self):
        """Test dictionary comprehension"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        results = {key: ov.pp.normalize(val) for key, val in data.items()}
        ```
        """)

        code = agent._extract_python_code(response)
        assert "{key: ov.pp.normalize(val)" in code
        ast.parse(code)

    def test_generator_expression(self):
        """Test generator expression"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        gen = (ov.pp.normalize(x) for x in adata_list)
        results = list(gen)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "(ov.pp.normalize(x) for x in adata_list)" in code
        ast.parse(code)

    def test_context_manager(self):
        """Test context manager (with statement)"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        with ov.batch_processor() as processor:
            adata = processor.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "with ov.batch_processor()" in code
        ast.parse(code)

    def test_exception_handling(self):
        """Test try/except/finally"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        try:
            adata = ov.pp.normalize(adata)
        except ValueError as e:
            print(f"Error: {e}")
            adata = ov.pp.qc(adata)
        finally:
            print("Processing complete")
        ```
        """)

        code = agent._extract_python_code(response)
        assert "try:" in code
        assert "except ValueError" in code
        assert "finally:" in code
        ast.parse(code)

    def test_multiline_strings(self):
        """Test multiline strings"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent('''
        ```python
        import omicverse as ov

        description = """
        This is a multiline
        description of the analysis
        """

        adata = ov.pp.normalize(adata)
        ```
        ''')

        code = agent._extract_python_code(response)
        assert '"""' in code
        ast.parse(code)

    def test_f_strings(self):
        """Test f-string formatting - may be transformed to string concatenation by ProactiveCodeTransformer"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        method = "normalize"
        print(f"Applying {method} to data")
        adata = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        # ProactiveCodeTransformer may convert f-strings to string concatenation for robustness
        # Accept either the original f-string or the transformed version
        assert ('f"Applying {method}' in code or
                ('Applying' in code and 'method' in code))
        # Result should still be valid Python
        ast.parse(code)


class TestInlineCodeExtraction:
    """Test extraction of inline code (not in fenced blocks)"""

    def test_simple_inline(self):
        """Test simple inline code extraction"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        To normalize your data, use:

        import omicverse as ov
        adata = ov.pp.normalize(adata)

        This will normalize the counts.
        """)

        code = agent._extract_python_code(response)
        assert "import omicverse" in code
        assert "ov.pp.normalize" in code
        ast.parse(code)

    def test_inline_with_comments(self):
        """Test inline code with comments"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        Here's how to process your data:

        # Import the library
        import omicverse as ov

        # Normalize the data
        adata = ov.pp.normalize(adata)
        """)

        code = agent._extract_python_code(response)
        assert "# Import" in code
        assert "# Normalize" in code
        ast.parse(code)

    def test_inline_mixed_with_text(self):
        """Test inline code mixed with explanatory text"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        First, import the library:
        import omicverse as ov

        Next, normalize your data:
        adata = ov.pp.normalize(adata)

        Finally, cluster the cells:
        adata = ov.pp.cluster(adata)
        """)

        code = agent._extract_python_code(response)
        # Should extract Python lines only
        assert "import omicverse" in code
        assert "ov.pp.normalize" in code
        assert "ov.pp.cluster" in code
        # Should NOT contain explanatory text
        assert "First, import" not in code
        assert "Next, normalize" not in code
        ast.parse(code)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_no_code_in_response(self):
        """Test response with no code - returns fallback code for robustness"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        I cannot help with that request. Please provide more information.
        """)

        # New behavior: returns fallback code instead of raising error for robustness
        code = agent._extract_python_code(response)
        # Should return valid fallback code
        assert "import omicverse as ov" in code or "import scanpy as sc" in code
        ast.parse(code)

    def test_invalid_syntax_in_fenced_block(self):
        """Test fenced block with invalid syntax - returns fallback code for robustness"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov
        this is not valid python syntax!!!
        ```
        """)

        # New behavior: returns fallback code instead of raising error for robustness
        code = agent._extract_python_code(response)
        # Should return valid fallback code
        assert "import omicverse as ov" in code or "import scanpy as sc" in code
        ast.parse(code)

    def test_empty_fenced_block(self):
        """Test response with no extractable code - returns fallback code for robustness"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        # Response with no code blocks at all
        response = "I cannot help with that."

        # New behavior: returns fallback code instead of raising error for robustness
        code = agent._extract_python_code(response)
        # Should return valid fallback code
        assert "import omicverse as ov" in code or "import scanpy as sc" in code
        ast.parse(code)

    def test_only_comments(self):
        """Test code with only comments"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        # This is a comment
        # Another comment
        ```
        """)

        # Comments are valid Python, should extract successfully
        code = agent._extract_python_code(response)
        assert "# This is a comment" in code
        assert "# Another comment" in code
        # Should be valid Python (comments are valid)
        ast.parse(code)

    def test_indentation_preserved(self):
        """Test that indentation is preserved for code blocks"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        if True:
            adata = ov.pp.normalize(adata)
            if True:
                adata = ov.pp.qc(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        # Check indentation is valid
        ast.parse(code)
        # Check nested if statements
        assert "if True:" in code

    def test_unicode_in_code(self):
        """Test code with unicode characters"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov

        # Process データ
        adata = ov.pp.normalize(adata)
        print("Finished ✓")
        ```
        """)

        code = agent._extract_python_code(response)
        assert "データ" in code
        assert "✓" in code
        ast.parse(code)

    def test_import_auto_injection(self):
        """Test automatic omicverse import injection"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        # No import statement
        adata = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        # Should auto-inject import
        assert "import omicverse as ov" in code
        ast.parse(code)

    def test_existing_import_not_duplicated(self):
        """Test that existing import is not duplicated"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov
        adata = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        # Should not duplicate import
        import_count = code.count("import omicverse as ov")
        assert import_count == 1
        ast.parse(code)


class TestCodeNormalization:
    """Test code normalization methods"""

    def test_dedent_indented_code(self):
        """Test dedenting of indented code blocks"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        # Code with extra indentation
        code = """
            import omicverse as ov
            adata = ov.pp.normalize(adata)
        """

        normalized = agent._normalize_code_candidate(code)
        # Should be dedented
        assert not normalized.startswith("    ")
        # Should still parse
        ast.parse(normalized)

    def test_strip_whitespace(self):
        """Test stripping of leading/trailing whitespace"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        code = "\n\n\nimport omicverse as ov\n\n\n"

        normalized = agent._normalize_code_candidate(code)
        # Should be stripped
        assert not normalized.startswith("\n")
        assert not normalized.endswith("\n\n")
        ast.parse(normalized)

    def test_empty_code_raises_error(self):
        """Test that empty code raises error"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        code = "   \n\n   \n   "

        with pytest.raises(ValueError, match="empty code candidate"):
            agent._normalize_code_candidate(code)


class TestGatherCodeCandidates:
    """Test code candidate gathering"""

    def test_fenced_blocks_prioritized(self):
        """Test that fenced blocks are prioritized over inline"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        Some inline code here:
        import omicverse as ov

        And a fenced block:
        ```python
        import omicverse as ov
        adata = ov.pp.normalize(adata)
        ```
        """)

        candidates = agent._gather_code_candidates(response)
        # Should only return fenced block (prioritized)
        assert len(candidates) >= 1
        # First candidate should be from fenced block
        assert "normalize" in candidates[0]

    def test_multiple_fenced_blocks_collected(self):
        """Test that all fenced blocks are collected"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov
        code1 = True
        ```

        ```python
        import omicverse as ov
        code2 = True
        ```
        """)

        candidates = agent._gather_code_candidates(response)
        # Should collect both blocks
        assert len(candidates) >= 2

    def test_inline_fallback(self):
        """Test inline extraction when no fenced blocks"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        import omicverse as ov
        adata = ov.pp.normalize(adata)
        """)

        candidates = agent._gather_code_candidates(response)
        assert len(candidates) >= 1
        assert "import omicverse" in candidates[0]


class TestFuzzingCodeExtraction:
    """Fuzzing tests for code extraction robustness"""

    def test_malformed_backticks(self):
        """Test handling of malformed backticks"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ``python
        import omicverse as ov
        adata = ov.pp.normalize(adata)
        ```
        """)

        # Should handle gracefully
        try:
            code = agent._extract_python_code(response)
            # If it extracts something, it should be valid
            ast.parse(code)
        except ValueError:
            # Or it should raise a clear error
            pass

    def test_nested_code_blocks(self):
        """Test nested code block markers (edge case)"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent("""
        ```python
        import omicverse as ov
        # Example: ```nested backticks```
        adata = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "import omicverse" in code
        ast.parse(code)

    def test_extremely_long_code(self):
        """Test handling of very long code blocks"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        # Generate long code
        lines = ["import omicverse as ov"]
        for i in range(1000):
            lines.append(f"step_{i} = ov.pp.normalize(adata)")

        code_block = "\n".join(lines)
        response = f"```python\n{code_block}\n```"

        code = agent._extract_python_code(response)
        assert "import omicverse" in code
        # Should still be valid
        ast.parse(code)

    def test_special_characters_in_strings(self):
        """Test code with special characters in strings"""
        agent = OmicVerseAgent(model="gpt-4o", api_key="test-key")

        response = textwrap.dedent(r"""
        ```python
        import omicverse as ov

        pattern = r"^[A-Z]+\d+\s*$"
        text = "Line with \"quotes\" and 'apostrophes'"
        adata = ov.pp.normalize(adata)
        ```
        """)

        code = agent._extract_python_code(response)
        assert "pattern" in code
        # Check for the word "quotes" (escaped quotes are preserved in raw strings)
        assert "quotes" in code
        assert "apostrophes" in code
        ast.parse(code)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

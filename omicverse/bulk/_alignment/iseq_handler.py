"""
iSeq Handler – process vendor-supplied FASTQ data.
Supports sample ID assignment, validation, and formatting for local FASTQ files.
"""
from __future__ import annotations

import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence
import logging

logger = logging.getLogger(__name__)

class ISeqHandler:
    """Handle vendor-provided FASTQ data."""

    def __init__(self,
                 sample_id_pattern: str = "auto",
                 paired_end: bool = True,
                 validate_files: bool = True):
        """
        Initialize the iSeq handler.

        Args:
            sample_id_pattern: How to derive sample IDs: "auto", "filename", or "directory".
            paired_end: Whether the reads are paired-end.
            validate_files: Whether to validate file integrity.
        """
        self.sample_id_pattern = sample_id_pattern
        self.paired_end = paired_end
        self.validate_files = validate_files

    def discover_fastq_files(self,
                           input_path: str | Path,
                           file_extensions: List[str] = None) -> List[Path]:
        """
        Discover FASTQ files under the given path.

        Args:
            input_path: File or directory to inspect.
            file_extensions: List of extensions to match, defaulting to [".fq", ".fastq", ".fq.gz", ".fastq.gz"].

        Returns:
            List of FASTQ file paths.
        """
        if file_extensions is None:
            file_extensions = [".fq", ".fastq", ".fq.gz", ".fastq.gz"]

        input_path = Path(input_path)
        fastq_files = []

        if input_path.is_file():
            # Single-file input.
            if any(str(input_path).endswith(ext) for ext in file_extensions):
                fastq_files.append(input_path)
        elif input_path.is_dir():
            # Recursively search the directory.
            for ext in file_extensions:
                fastq_files.extend(input_path.rglob(f"*{ext}"))
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if not fastq_files:
            raise ValueError(f"No FASTQ files found in {input_path}")

        # Sort to maintain consistent ordering.
        fastq_files.sort()
        logger.info(f"Discovered {len(fastq_files)} FASTQ files")
        return fastq_files

    def extract_sample_id(self, file_path: Path) -> str:
        """
        Extract a sample ID from the given file path.

        Args:
            file_path: FASTQ file path.

        Returns:
            Sample ID string.
        """
        if self.sample_id_pattern == "auto":
            return self._auto_extract_sample_id(file_path)
        elif self.sample_id_pattern == "filename":
            return self._extract_from_filename(file_path)
        elif self.sample_id_pattern == "directory":
            return self._extract_from_directory(file_path)
        else:
            raise ValueError(f"Unknown pattern: {self.sample_id_pattern}")

    def _auto_extract_sample_id(self, file_path: Path) -> str:
        """Automatically derive a sample ID."""
        # Special case: honour explicit SRA-style IDs (SRR/ERR/DRR) anywhere
        # in the filename, so that FASTQ-based workflows stay consistent with
        # GEO/SRA-based naming (e.g. "SRR123456.fastq.gz" → "SRR123456").
        sra_match = re.search(r"(SRR\d+|ERR\d+|DRR\d+)", file_path.name)
        if sra_match:
            return sra_match.group(1)

        # Generic heuristics for vendor-style filenames.
        filename = file_path.stem  # Remove only the last extension.

        # Pattern 1: remove common sequencing suffixes (_R1, _R2, .R1, .R2, _1, _2).
        cleaned = re.sub(r'[._][Rr]?[12]$', '', filename)

        # Pattern 2: if underscores remain, take the first segment.
        if '_' in cleaned:
            cleaned = cleaned.split('_')[0]

        # Pattern 3: verify the cleaned ID matches common formatting.
        if re.match(r'^[A-Za-z0-9_-]+$', cleaned):
            return cleaned
        else:
            # Fallback: sanitize the filename by removing non-alphanumeric characters.
            return re.sub(r'[^A-Za-z0-9_-]', '', cleaned)

    def _extract_from_filename(self, file_path: Path) -> str:
        """Extract a sample ID from the filename."""
        filename = file_path.stem
        # Strip sequencing suffixes.
        return re.sub(r'[._][Rr]?[12]$', '', filename)

    def _extract_from_directory(self, file_path: Path) -> str:
        """Extract a sample ID from the parent directory."""
        return file_path.parent.name

    def group_paired_end(self, fastq_files: List[Path]) -> List[Tuple[str, Path, Optional[Path]]]:
        """
        Group FASTQ files into paired-end tuples.

        Args:
            fastq_files: List of FASTQ file paths.

        Returns:
            List of pairings: [(sample_id, R1_path, R2_path), ...].
        """
        if not self.paired_end:
            # Handle single-end sequencing.
            return [(self.extract_sample_id(fq), fq, None) for fq in fastq_files]

        # Paired-end grouping.
        sample_groups: Dict[str, List[Path]] = {}

        for fq_file in fastq_files:
            sample_id = self.extract_sample_id(fq_file)
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(fq_file)

        # Validate pairings.
        paired_results = []
        for sample_id, files in sample_groups.items():
            if len(files) == 1:
                # Single file—either single-end or missing R2.
                logger.warning(f"Sample {sample_id} has only one file: {files[0]}")
                paired_results.append((sample_id, files[0], None))
            elif len(files) == 2:
                # Two files—identify R1 and R2.
                r1_file, r2_file = self._identify_r1_r2(files)
                paired_results.append((sample_id, r1_file, r2_file))
            else:
                # More than two files—log a warning and use the first two.
                logger.warning(f"Sample {sample_id} has {len(files)} files, using first two")
                r1_file, r2_file = self._identify_r1_r2(files[:2])
                paired_results.append((sample_id, r1_file, r2_file))

        return paired_results

    def _identify_r1_r2(self, files: List[Path]) -> Tuple[Path, Path]:
        """Identify R1 and R2 files."""
        # Use filename patterns when available.
        r1_candidates = [f for f in files if re.search(r'[._][Rr]1[._]', str(f))]
        r2_candidates = [f for f in files if re.search(r'[._][Rr]2[._]', str(f))]

        if len(r1_candidates) == 1 and len(r2_candidates) == 1:
            return r1_candidates[0], r2_candidates[0]

        # Without explicit markers, sort alphabetically.
        files_sorted = sorted(files, key=lambda x: str(x))
        if len(files_sorted) >= 2:
            return files_sorted[0], files_sorted[1]

        # Final fallback.
        return files[0], files[1] if len(files) > 1 else None

    def validate_fastq_files(self, fastq_pairs: List[Tuple[str, Path, Optional[Path]]]) -> bool:
        """
        Validate the supplied FASTQ files.

        Args:
            fastq_pairs: Paired FASTQ file list.

        Returns:
            True when all files pass validation.
        """
        all_valid = True

        for sample_id, r1_path, r2_path in fastq_pairs:
            # Validate the R1 file.
            if not self._validate_single_file(r1_path):
                logger.error(f"Invalid R1 file for sample {sample_id}: {r1_path}")
                all_valid = False
                continue

            # Validate the R2 file when present.
            if r2_path and not self._validate_single_file(r2_path):
                logger.error(f"Invalid R2 file for sample {sample_id}: {r2_path}")
                all_valid = False
                continue

            logger.info(f"Validated sample {sample_id}: R1={r1_path}, R2={r2_path}")

        return all_valid

    def _validate_single_file(self, file_path: Path) -> bool:
        """Validate a single FASTQ file."""
        if not file_path.exists():
            return False

        if file_path.stat().st_size == 0:
            return False

        # Basic format check: inspect the first record header.
        try:
            if str(file_path).endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline().strip()
            else:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()

            # FASTQ records should start with '@'.
            return first_line.startswith('@')
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    def create_sample_metadata(self,
                             fastq_pairs: List[Tuple[str, Path, Optional[Path]]],
                             output_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Build a sample metadata table.

        Args:
            fastq_pairs: Paired FASTQ file list.
            output_file: Optional path for writing the CSV output.

        Returns:
            Sample metadata DataFrame.
        """
        metadata = []

        for sample_id, r1_path, r2_path in fastq_pairs:
            meta = {
                'sample_id': sample_id,
                'r1_path': str(r1_path),
                'r2_path': str(r2_path) if r2_path else None,
                'paired_end': r2_path is not None,
                'file_size_r1_mb': r1_path.stat().st_size / (1024 * 1024),
                'file_size_r2_mb': r2_path.stat().st_size / (1024 * 1024) if r2_path else 0,
            }
            metadata.append(meta)

        df = pd.DataFrame(metadata)

        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Sample metadata saved to {output_file}")

        return df

    def process_company_data(self,
                           input_path: str | Path,
                           output_dir: str | Path,
                           sample_prefix: str = "Sample") -> Dict[str, Any]:
        """
        Main entry point for processing vendor FASTQ data.

        Args:
            input_path: FASTQ file or directory path.
            output_dir: Output directory.
            sample_prefix: Sample ID prefix.

        Returns:
            Result dictionary containing processed metadata.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing company FASTQ data from {input_path}")

        # 1. Discover FASTQ files.
        fastq_files = self.discover_fastq_files(input_path)

        # 2. Pair up the FASTQ files.
        fastq_pairs = self.group_paired_end(fastq_files)

        # 3. Validate files.
        if self.validate_files:
            if not self.validate_fastq_files(fastq_pairs):
                raise ValueError("Some FASTQ files failed validation")

        # 4. Build sample metadata.
        metadata_file = output_dir / "sample_metadata.csv"
        metadata_df = self.create_sample_metadata(fastq_pairs, metadata_file)

        # 5. Create standardized sample ID mappings.
        sample_id_mapping = {}
        for i, (original_id, r1_path, r2_path) in enumerate(fastq_pairs):
            standardized_id = f"{sample_prefix}_{i+1:03d}"
            sample_id_mapping[standardized_id] = {
                'original_id': original_id,
                'r1_path': str(r1_path),
                'r2_path': str(r2_path) if r2_path else None,
                'paired_end': r2_path is not None
            }

        # 6. Persist the mapping file.
        mapping_file = output_dir / "sample_id_mapping.json"
        import json
        with open(mapping_file, 'w') as f:
            json.dump(sample_id_mapping, f, indent=2)

        logger.info(f"Processed {len(fastq_pairs)} samples")
        logger.info(f"Results saved to {output_dir}")

        return {
            'sample_pairs': fastq_pairs,
            'metadata_df': metadata_df,
            'sample_id_mapping': sample_id_mapping,
            'metadata_file': metadata_file,
            'mapping_file': mapping_file,
            'output_dir': output_dir
        }

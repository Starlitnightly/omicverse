"""Fixed GEO API methods."""

def get_series_matrix(self, gse_accession: str) -> Dict[str, Any]:
    """Get series matrix file.
    
    Args:
        gse_accession: GEO Series accession (e.g., "GSE1234")
    
    Returns:
        Series matrix data
    """
    url = f"{self.geo_base_url}/acc.cgi"
    params = {
        "acc": gse_accession,
        "targ": "self",
        "form": "text",
        "view": "full"
    }
    
    response = self.session.get(url, params=params)
    response.raise_for_status()
    
    parsed = self._parse_soft_format(response.text)
    
    # Ensure series section exists
    if 'series' not in parsed and parsed:
        # If there's no explicit series section, wrap everything in it
        first_key = list(parsed.keys())[0] if parsed else 'data'
        parsed = {'series': parsed.get(first_key, parsed)}
    
    return parsed

def get_platform_data(self, gpl_accession: str) -> Dict[str, Any]:
    """Get platform annotation data.
    
    Args:
        gpl_accession: GEO Platform accession (e.g., "GPL570")
    
    Returns:
        Platform information
    """
    url = f"{self.geo_base_url}/acc.cgi"
    params = {
        "acc": gpl_accession,
        "targ": "self",
        "form": "text",
        "view": "brief"
    }
    
    response = self.session.get(url, params=params)
    response.raise_for_status()
    
    parsed = self._parse_soft_format(response.text)
    
    # Ensure platform section exists
    if 'platform' not in parsed and parsed:
        # If there's no explicit platform section, wrap everything in it
        first_key = list(parsed.keys())[0] if parsed else 'data'
        parsed = {'platform': parsed.get(first_key, parsed)}
    
    return parsed

def get_expression_data(
    self,
    gse_accession: str,
    sample_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get expression data for a series.
    
    Args:
        gse_accession: GEO Series accession
        sample_ids: Optional list of sample IDs to include
    
    Returns:
        Expression data matrix
    """
    # Get the series matrix
    matrix_data = self.get_series_matrix(gse_accession)
    series_data = matrix_data.get('series', matrix_data)
    
    # Extract expression values
    expression_data = {
        "series_id": gse_accession,
        "title": series_data.get("series_title", ""),
        "platform": series_data.get("series_platform_id", ""),
        "samples": {},
        "features": []
    }
    
    # Parse sample data if available
    if "sample_geo_accession" in series_data:
        sample_list = series_data["sample_geo_accession"]
        if isinstance(sample_list, str):
            sample_list = [sample_list]
        
        # Also get sample titles if available
        sample_titles = series_data.get("sample_title", [])
        if isinstance(sample_titles, str):
            sample_titles = [sample_titles]
        
        if sample_ids:
            # Filter to requested samples
            indices = [i for i, s in enumerate(sample_list) if s in sample_ids]
            sample_list = [sample_list[i] for i in indices]
            if sample_titles:
                sample_titles = [sample_titles[i] for i in indices if i < len(sample_titles)]
        
        for i, sample_id in enumerate(sample_list):
            sample_data = {}
            if i < len(sample_titles):
                sample_data["sample_title"] = sample_titles[i]
            expression_data["samples"][sample_id] = sample_data
    
    return expression_data

def _parse_soft_format(self, text: str) -> Dict[str, Any]:
    """Parse SOFT format text into structured data.
    
    Args:
        text: SOFT format text
    
    Returns:
        Parsed data dictionary
    """
    data = {}
    current_section = None
    current_subsection = {}
    
    for line in text.split('\n'):
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('^'):
            # New section
            if current_section and current_subsection:
                data[current_section] = current_subsection
            current_section = line[1:].lower()
            current_subsection = {}
        
        elif line.startswith('!'):
            # Metadata line - handle both tab and = separators
            if '\t' in line:
                parts = line[1:].split('\t')
                key = parts[0].lower().replace(' ', '_')
                values = [v.strip('"') for v in parts[1:] if v]
                
                if len(values) == 1:
                    current_subsection[key] = values[0]
                elif len(values) > 1:
                    current_subsection[key] = values
            elif ' = ' in line:
                key, value = line[1:].split(' = ', 1)
                key = key.lower().replace(' ', '_')
                
                # Handle multi-value fields
                if key in current_subsection:
                    if not isinstance(current_subsection[key], list):
                        current_subsection[key] = [current_subsection[key]]
                    current_subsection[key].append(value)
                else:
                    current_subsection[key] = value
    
    # Save last section
    if current_section and current_subsection:
        data[current_section] = current_subsection
    elif current_subsection and not current_section:
        # If we have data but no section was defined, use a default section
        data = current_subsection
    
    return data

def download_supplementary_files(self, gse_accession: str) -> List[str]:
    """Get URLs for supplementary files.
    
    Args:
        gse_accession: GEO Series accession
    
    Returns:
        List of supplementary file URLs
    """
    # Parse HTML to find actual file links
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_accession}"
    
    try:
        response = self.session.get(url)
        response.raise_for_status()
        
        # Simple regex to find FTP links
        import re
        ftp_pattern = r'(ftp://[^\s<>"]+(?:\.tar|\.txt\.gz|\.gz|\.zip))'
        urls = re.findall(ftp_pattern, response.text)
        
        # Also add the direct download link
        urls.append(f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_accession}&format=file")
        
        return list(set(urls))  # Remove duplicates
    except:
        # Fallback to standard FTP patterns
        gse_prefix = gse_accession[:6] + "nnn"
        return [
            f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_accession}/suppl/",
            f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_accession}&format=file"
        ]
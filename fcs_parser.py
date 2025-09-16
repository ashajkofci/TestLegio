#!/usr/bin/env python3
"""
Basic FCS file parser for flow cytometry data.
Implements minimal parsing functionality for FCS 3.1 format files.
"""

import struct
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


class FCSParser:
    """Basic FCS file parser for flow cytometry data."""
    
    def __init__(self):
        self.header = {}
        self.metadata = {}
        self.data = None
        
    def parse_fcs_file(self, filename: str) -> pd.DataFrame:
        """
        Parse an FCS file and return data as pandas DataFrame.
        
        Args:
            filename: Path to FCS file
            
        Returns:
            DataFrame with flow cytometry data
        """
        with open(filename, 'rb') as f:
            # Read FCS header (first 58 bytes)
            header_data = f.read(58)
            
            # Parse header
            fcs_version = header_data[:10].decode('ascii').strip()
            if not fcs_version.startswith('FCS'):
                raise ValueError(f"Not an FCS file: {filename}")
                
            # Extract segment offsets from header
            text_start = int(header_data[10:18].decode('ascii').strip())
            text_end = int(header_data[18:26].decode('ascii').strip())
            data_start = int(header_data[26:34].decode('ascii').strip())
            data_end = int(header_data[34:42].decode('ascii').strip())
            
            # Read TEXT segment
            f.seek(text_start)
            text_data = f.read(text_end - text_start + 1).decode('ascii', errors='ignore')
            
            # Parse metadata from TEXT segment
            metadata = self._parse_text_segment(text_data)
            
            # Read DATA segment
            f.seek(data_start)
            data_size = data_end - data_start + 1
            data_bytes = f.read(data_size)
            
            # Parse data based on metadata
            data_array = self._parse_data_segment(data_bytes, metadata)
            
            # Create DataFrame with parameter names
            param_names = self._get_parameter_names(metadata)
            df = pd.DataFrame(data_array, columns=param_names)
            
            # Store metadata for reference
            self.metadata = metadata
            
            return df
    
    def _parse_text_segment(self, text_data: str) -> Dict[str, str]:
        """Parse the TEXT segment to extract metadata."""
        metadata = {}
        
        # The first character is the delimiter
        if not text_data:
            return metadata
            
        delimiter = text_data[0]
        
        # Split by delimiter and parse key-value pairs
        parts = text_data[1:].split(delimiter)
        
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                key = parts[i].strip()
                value = parts[i + 1].strip()
                metadata[key] = value
                
        return metadata
    
    def _parse_data_segment(self, data_bytes: bytes, metadata: Dict[str, str]) -> np.ndarray:
        """Parse the DATA segment to extract numerical data."""
        # Get data format information
        datatype = metadata.get('$DATATYPE', 'I')  # Default to integer
        par = int(metadata.get('$PAR', '0'))  # Number of parameters
        tot = int(metadata.get('$TOT', '0'))  # Number of events
        
        if datatype == 'I':  # Integer data
            # Assume 32-bit integers (4 bytes per value)
            bytes_per_value = 4
            format_char = 'I'  # Unsigned integer
        elif datatype == 'F':  # Float data
            bytes_per_value = 4
            format_char = 'f'  # Float
        else:
            raise ValueError(f"Unsupported data type: {datatype}")
        
        # Calculate expected size
        expected_size = par * tot * bytes_per_value
        
        if len(data_bytes) < expected_size:
            raise ValueError(f"Data segment too small: {len(data_bytes)} < {expected_size}")
        
        # Unpack binary data
        format_string = f'<{par * tot}{format_char}'  # Little-endian
        try:
            values = struct.unpack(format_string, data_bytes[:expected_size])
        except struct.error as e:
            # Try with different byte order
            format_string = f'>{par * tot}{format_char}'  # Big-endian
            values = struct.unpack(format_string, data_bytes[:expected_size])
        
        # Reshape into matrix (events x parameters)
        data_array = np.array(values).reshape(tot, par)
        
        return data_array
    
    def _get_parameter_names(self, metadata: Dict[str, str]) -> list:
        """Extract parameter names from metadata."""
        par = int(metadata.get('$PAR', '0'))
        param_names = []
        
        for i in range(1, par + 1):
            # Try to get parameter name, fall back to generic name
            param_name = metadata.get(f'$P{i}N', f'P{i}')
            if not param_name or param_name == ' ':
                param_name = f'Parameter_{i}'
            param_names.append(param_name)
            
        return param_names
    
    def get_metadata(self) -> Dict[str, str]:
        """Return the parsed metadata."""
        return self.metadata.copy()


def load_fcs_data(filename: str) -> pd.DataFrame:
    """
    Convenience function to load FCS data.
    
    Args:
        filename: Path to FCS file
        
    Returns:
        DataFrame with flow cytometry data
    """
    parser = FCSParser()
    return parser.parse_fcs_file(filename)


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            df = load_fcs_data(filename)
            print(f"Loaded FCS file: {filename}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
            print("\nData types:")
            print(df.dtypes)
            print("\nSummary statistics:")
            print(df.describe())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print("Usage: python fcs_parser.py <fcs_file>")
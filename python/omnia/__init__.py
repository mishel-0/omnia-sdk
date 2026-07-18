"""
Omnia SDK — Python API
pip install omnia-sdk
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple

__version__ = "2.0.0"


class OmniaError(Exception):
    """Base exception for Omnia SDK errors."""
    pass


class SliceInfo:
    """Metadata for a single CT slice."""
    def __init__(self, rows: int, cols: int, patient_id: str = "",
                 study_uid: str = "", sop_uid: str = ""):
        self.rows = rows
        self.cols = cols
        self.patient_id = patient_id
        self.study_uid = study_uid
        self.sop_uid = sop_uid

    def __repr__(self):
        return (f"SliceInfo({self.rows}x{self.cols}, "
                f"patient={self.patient_id[:20]})")


class OmniaCompressor:
    """
    High-level Python API for .omnia compression.
    
    Usage:
        comp = OmniaCompressor()
        result = comp.compress("/path/to/ct/slices/", "study.omnia")
        comp.decompress("study.omnia", "/path/to/restored/")
    """
    
    def __init__(self, mode: str = "lossless"):
        """Initialize compressor.
        
        Args:
            mode: "lossless" (default) or "lossy"
        """
        self.mode = mode
        self._bundler_path = None
    
    def compress(self, input_dir: str, output_path: str,
                 verbose: bool = False) -> Dict:
        """Compress a directory of DICOM slices to .omnia.
        
        Args:
            input_dir: Directory containing .dcm files
            output_path: Path for .omnia output file
            verbose: Print progress
            
        Returns:
            Dict with keys: success, original_size, compressed_size, ratio
        """
        from omnia_container import bundle_dicoms_to_omnia
        
        if not os.path.isdir(input_dir):
            raise OmniaError(f"Input directory not found: {input_dir}")
        
        try:
            bundle_dicoms_to_omnia(input_dir, output_path, verbose=verbose)
        except Exception as e:
            raise OmniaError(f"Compression failed: {e}")
        
        orig = sum(os.path.getsize(os.path.join(input_dir, f))
                   for f in os.listdir(input_dir)
                   if f.endswith('.dcm'))
        comp = os.path.getsize(output_path)
        
        return {
            "success": True,
            "original_size": orig,
            "compressed_size": comp,
            "ratio": orig / comp if comp > 0 else 0,
        }
    
    def decompress(self, input_path: str, output_dir: str,
                   verbose: bool = False) -> Dict:
        """Decompress .omnia back to DICOM slices.
        
        Args:
            input_path: Path to .omnia file
            output_dir: Directory for restored DICOMs
            verbose: Print progress
            
        Returns:
            Dict with keys: success, num_slices, output_dir
        """
        from omnia_container import unbundle_omnia_to_dicoms
        
        if not os.path.isfile(input_path):
            raise OmniaError(f"File not found: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            unbundle_omnia_to_dicoms(input_path, output_dir, verbose=verbose)
        except Exception as e:
            raise OmniaError(f"Decompression failed: {e}")
        
        num = len([f for f in os.listdir(output_dir) if f.endswith('.dcm')])
        
        return {
            "success": True,
            "num_slices": num,
            "output_dir": output_dir,
        }
    
    def info(self, input_path: str) -> List[SliceInfo]:
        """Get metadata from .omnia file.
        
        Args:
            input_path: Path to .omnia file
            
        Returns:
            List of SliceInfo objects
        """
        if not os.path.isfile(input_path):
            raise OmniaError(f"File not found: {input_path}")
        
        import struct
        import zstd
        import json
        
        MAGIC = b"OMI2"
        
        with open(input_path, "rb") as f:
            magic = f.read(4)
            if magic not in [b"OMI2"]:
                raise OmniaError(f"Invalid .omnia file")
            
            f.read(2)  # version + compression
            orig_size = struct.unpack("<Q", f.read(8))[0]
            pixel_size = struct.unpack("<Q", f.read(8))[0]
            
            meta_size = struct.unpack("<Q", f.read(8))[0]
            meta_comp = f.read(meta_size)
            meta = json.loads(zstd.decompress(meta_comp))
        
        slices = []
        for s in meta.get("slices", []):
            slices.append(SliceInfo(
                rows=s.get("rows", 512),
                cols=s.get("cols", 512),
                patient_id=s.get("0010_0020", ""),
                study_uid=s.get("0020_000D", ""),
                sop_uid=s.get("0008_0018", ""),
            ))
        
        return slices
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# ─── Convenience functions ──────────────────────────────────

def compress(input_dir: str, output_path: str, **kwargs) -> Dict:
    """One-shot compress."""
    return OmniaCompressor().compress(input_dir, output_path, **kwargs)


def decompress(input_path: str, output_dir: str, **kwargs) -> Dict:
    """One-shot decompress."""
    return OmniaCompressor().decompress(input_path, output_dir, **kwargs)


def info(input_path: str) -> List[SliceInfo]:
    """Get .omnia file info."""
    return OmniaCompressor().info(input_path)


def verify(input_path: str) -> bool:
    """Verify .omnia file integrity."""
    try:
        s = OmniaCompressor().info(input_path)
        return len(s) > 0
    except Exception:
        return False

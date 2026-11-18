"""Omnia SDK — Model Context Protocol (MCP) Server.

Enables AI agents (Claude, Cursor, Codex) to inspect, validate,
and convert Whole-Slide Images (.omnia and .svs) using omnia-sdk.
"""
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Omnia SDK",
    dependencies=["numpy", "zstandard", "mcp"]
)

@mcp.tool()
def get_sdk_status() -> dict:
    """Returns the Omnia SDK version and supported whole-slide image formats."""
    return {
        "status": "ready",
        "package": "omnia-sdk",
        "description": "PyTorch-native container for whole-slide images (13x faster than OpenSlide)",
        "supported_formats": [".omnia", ".svs", ".tiff", ".tif"],
        "features": [
            "Zstd lossless per-tile compression",
            "Zero-copy random access",
            "Pyramid multi-level reading",
            "SVS to Omnia conversion"
        ]
    }

@mcp.tool()
def inspect_slide(slide_path: str) -> dict:
    """Inspects a whole-slide image file (.omnia or .svs) and returns dimensions, tiles, and metadata."""
    path = Path(slide_path)
    if not path.exists():
        return {"error": f"Slide not found: {slide_path}"}

    suffix = path.suffix.lower()
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)

    if suffix == ".omnia":
        try:
            from omnia_sdk.container import OmniaReader
            with OmniaReader(str(path)) as reader:
                meta = reader.metadata
                return {
                    "format": "omnia",
                    "file_name": path.name,
                    "size_mb": size_mb,
                    "dimensions": [meta.width, meta.height],
                    "tile_size": meta.tile_size,
                    "num_tiles": len(reader),
                    "levels": meta.num_levels,
                    "compression": "zstd"
                }
        except Exception as e:
            return {"error": f"Failed to read .omnia container: {e}"}

    return {
        "format": "svs/tiff",
        "file_name": path.name,
        "size_mb": size_mb,
        "note": "Use convert_svs_to_omnia to convert to high-performance .omnia container"
    }

@mcp.tool()
def verify_slide_integrity(slide_path: str) -> dict:
    """Verifies format integrity of an .omnia slide container."""
    path = Path(slide_path)
    if not path.exists():
        return {"error": f"File not found: {slide_path}"}
    if path.suffix.lower() != ".omnia":
        return {"error": "Integrity check is only supported for .omnia files"}

    try:
        from omnia_sdk.container import OmniaReader
        with OmniaReader(str(path)) as reader:
            is_valid = reader.verify_integrity()
            return {
                "file_name": path.name,
                "integrity_valid": is_valid,
                "num_tiles_checked": len(reader)
            }
    except Exception as e:
        return {"error": f"Integrity check failed: {e}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")

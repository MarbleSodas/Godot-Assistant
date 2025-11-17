"""
File system tools for the planning agent.

Provides tools for reading files, listing directories, and searching codebase.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import aiofiles

from strands import tool


@tool
async def read_file(file_path: str) -> Dict[str, any]:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read (relative or absolute)

    Returns:
        Dictionary with status and file content or error message
    """
    try:
        # Resolve path
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            return {
                "status": "error",
                "content": [{"text": f"File not found: {file_path}"}]
            }

        # Check if it's a file (not a directory)
        if not path.is_file():
            return {
                "status": "error",
                "content": [{"text": f"Path is not a file: {file_path}"}]
            }

        # Read file content
        async with aiofiles.open(path, mode='r', encoding='utf-8', errors='ignore') as f:
            content = await f.read()

        return {
            "status": "success",
            "content": [{
                "text": f"File: {file_path}\n\n{content}"
            }]
        }

    except PermissionError:
        return {
            "status": "error",
            "content": [{"text": f"Permission denied: {file_path}"}]
        }
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error reading file: {str(e)}"}]
        }


@tool
async def list_files(directory: str = ".", pattern: str = "*") -> Dict[str, any]:
    """
    List files and directories in a specified directory.

    Args:
        directory: Directory path to list (default: current directory)
        pattern: Glob pattern to filter files (default: "*" for all files)

    Returns:
        Dictionary with status and list of files/directories
    """
    try:
        # Resolve directory path
        dir_path = Path(directory).resolve()

        # Check if directory exists
        if not dir_path.exists():
            return {
                "status": "error",
                "content": [{"text": f"Directory not found: {directory}"}]
            }

        # Check if it's a directory
        if not dir_path.is_dir():
            return {
                "status": "error",
                "content": [{"text": f"Path is not a directory: {directory}"}]
            }

        # List files matching pattern
        files = []
        directories = []

        for item in dir_path.glob(pattern):
            if item.is_file():
                files.append(str(item.relative_to(dir_path)))
            elif item.is_dir():
                directories.append(str(item.relative_to(dir_path)))

        # Format output
        output = f"Directory: {directory}\n\n"

        if directories:
            output += "Directories:\n"
            for d in sorted(directories):
                output += f"  📁 {d}/\n"
            output += "\n"

        if files:
            output += "Files:\n"
            for f in sorted(files):
                output += f"  📄 {f}\n"

        if not directories and not files:
            output += "No files or directories found matching the pattern.\n"

        return {
            "status": "success",
            "content": [{"text": output}]
        }

    except PermissionError:
        return {
            "status": "error",
            "content": [{"text": f"Permission denied: {directory}"}]
        }
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error listing directory: {str(e)}"}]
        }


@tool
async def search_codebase(
    pattern: str,
    directory: str = ".",
    file_pattern: str = "*.py",
    max_results: int = 50
) -> Dict[str, any]:
    """
    Search for a pattern in the codebase using regular expressions.

    Args:
        pattern: Regular expression pattern to search for
        directory: Directory to search in (default: current directory)
        file_pattern: Glob pattern for files to search (default: "*.py")
        max_results: Maximum number of results to return (default: 50)

    Returns:
        Dictionary with status and search results
    """
    try:
        # Resolve directory path
        dir_path = Path(directory).resolve()

        # Check if directory exists
        if not dir_path.exists():
            return {
                "status": "error",
                "content": [{"text": f"Directory not found: {directory}"}]
            }

        # Compile regex pattern
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            return {
                "status": "error",
                "content": [{"text": f"Invalid regex pattern: {str(e)}"}]
            }

        # Search files
        results = []
        files_searched = 0

        for file_path in dir_path.rglob(file_pattern):
            if not file_path.is_file():
                continue

            files_searched += 1

            try:
                async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()

                # Find all matches in the file
                for line_num, line in enumerate(content.split('\n'), 1):
                    if regex.search(line):
                        results.append({
                            "file": str(file_path.relative_to(dir_path)),
                            "line": line_num,
                            "content": line.strip()
                        })

                        # Stop if we've reached max results
                        if len(results) >= max_results:
                            break

            except Exception as e:
                # Skip files that can't be read
                continue

            if len(results) >= max_results:
                break

        # Format output
        if not results:
            output = f"No matches found for pattern: {pattern}\n"
            output += f"Searched {files_searched} files in {directory}"
        else:
            output = f"Found {len(results)} matches for pattern: {pattern}\n"
            output += f"(Searched {files_searched} files)\n\n"

            for i, result in enumerate(results, 1):
                output += f"{i}. {result['file']}:{result['line']}\n"
                output += f"   {result['content']}\n\n"

            if len(results) >= max_results:
                output += f"\nNote: Results limited to {max_results} matches.\n"

        return {
            "status": "success",
            "content": [{"text": output}]
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error searching codebase: {str(e)}"}]
        }

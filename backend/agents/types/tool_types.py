"""
Shared type definitions for agent tools.

This module provides common type definitions used across different tool modules
to ensure type safety and consistency.
"""

from typing import Dict, List, TypedDict, Union, Literal, Optional, Any
from enum import Enum


class ToolStatus(str, Enum):
    """Enumeration for tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ContentBlock(TypedDict):
    """Type definition for content blocks returned by tools."""
    text: str


class ToolResponse(TypedDict):
    """Standardized response structure for all tools."""
    status: ToolStatus
    content: List[ContentBlock]
    metadata: Optional[Dict[str, Any]]


class SearchResult(TypedDict):
    """Type definition for search results."""
    file: str
    line: int
    content: str


class FileInfo(TypedDict):
    """Type definition for file information."""
    path: str
    name: str
    size: int
    is_file: bool
    is_directory: bool
    modified_time: Optional[float]


class DirectoryListing(TypedDict):
    """Type definition for directory listing results."""
    directory: str
    files: List[str]
    directories: List[str]
    total_count: int


# Common error response types
class ErrorResponse(TypedDict):
    """Type definition for error responses."""
    status: Literal[ToolStatus.ERROR]
    content: List[ContentBlock]
    error_type: str
    error_details: Optional[str]


class SuccessResponse(TypedDict):
    """Type definition for success responses."""
    status: Literal[ToolStatus.SUCCESS]
    content: List[ContentBlock]
    metadata: Optional[Dict[str, Any]]


# Union type for all possible tool responses
ToolResult = Union[SuccessResponse, ErrorResponse]


# Type aliases for better readability
FilePath = str
DirectoryName = str
SearchPattern = str
FilePattern = str
MaxResults = int
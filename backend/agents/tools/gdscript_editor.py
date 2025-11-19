"""
GDScript Editor for Advanced Script Manipulation.

This module provides specialized tools for analyzing, editing, and manipulating
GDScript files with syntax validation, code completion, and structural analysis.
"""

import ast
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

from strands import tool
from .file_tools import FileTools, GDScriptEditResult

logger = logging.getLogger(__name__)


@dataclass
class GDScriptMethodInfo:
    """Information about a GDScript method."""
    name: str
    parameters: List[Tuple[str, str]]  # (name, type) pairs
    return_type: Optional[str]
    is_static: bool
    is_virtual: bool
    is_override: bool
    body: str
    start_line: int
    end_line: int
    decorators: List[str]


@dataclass
class GDScriptClassInfo:
    """Information about a GDScript class."""
    name: Optional[str]  # None for implicit class
    extends: Optional[str]
    methods: List[GDScriptMethodInfo]
    properties: List[Tuple[str, str, Any]]  # (name, type, value)
    constants: List[Tuple[str, str, Any]]  # (name, type, value)
    signals: List[Tuple[str, List[str]]]  # (name, parameters)
    inner_classes: List['GDScriptClassInfo']
    imports: List[str]
    start_line: int
    end_line: int


@dataclass
class GDScriptValidationResult:
    """Result of GDScript validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class GDScriptEditor:
    """
    Advanced GDScript editor with syntax validation and structural analysis.

    This class provides comprehensive tools for manipulating GDScript files
    with proper understanding of GDScript syntax and structure.
    """

    def __init__(self):
        """Initialize GDScript editor."""
        self.file_tools = FileTools()

    # Analysis Methods
    async def analyze_gdscript(self, file_path: str) -> GDScriptClassInfo:
        """
        Analyze a GDScript file and extract structural information.

        Args:
            file_path: Path to the GDScript file

        Returns:
            GDScriptClassInfo with complete structural analysis
        """
        try:
            # Read the file
            file_result = await self.file_tools.read_file_safe(file_path)
            if not file_result.success:
                raise ValueError(f"Failed to read file: {file_result.error}")

            content = file_result.new_content
            lines = content.split('\n')

            # Parse the file structure
            class_info = self._parse_class_structure(content, lines)

            logger.info(f"Successfully analyzed GDScript: {file_path}")
            return class_info

        except Exception as e:
            logger.error(f"Error analyzing GDScript {file_path}: {e}")
            raise

    def _parse_class_structure(self, content: str, lines: List[str]) -> GDScriptClassInfo:
        """Parse the structure of a GDScript class."""
        # Extract class declaration
        class_name = None
        extends = None
        imports = []
        class_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Extract imports
            if stripped.startswith('extends '):
                extends = stripped[8:].strip()
            elif stripped.startswith('class_name '):
                class_name = stripped[11:].strip()
            elif stripped.startswith('extends preload('):
                # Handle preloaded extends
                extends = stripped[8:].strip()
            elif stripped.startswith('const ') and '=' in stripped:
                # Handle constants at file level
                continue
            elif stripped.startswith('var '):
                # Handle variables at file level
                continue
            elif stripped.startswith('func ') or stripped.startswith('static func '):
                # First method found, this is where the class starts
                class_start = i
                break

        # Parse methods
        methods = self._extract_methods(lines, class_start)

        # Parse properties and constants
        properties, constants = self._extract_properties_and_constants(lines, class_start)

        # Parse signals
        signals = self._extract_signals(lines, class_start)

        # Parse inner classes
        inner_classes = self._extract_inner_classes(lines, class_start)

        return GDScriptClassInfo(
            name=class_name,
            extends=extends,
            methods=methods,
            properties=properties,
            constants=constants,
            signals=signals,
            inner_classes=inner_classes,
            imports=imports,
            start_line=class_start,
            end_line=len(lines) - 1
        )

    def _extract_methods(self, lines: List[str], start_line: int) -> List[GDScriptMethodInfo]:
        """Extract all methods from the lines."""
        methods = []
        i = start_line

        while i < len(lines):
            line = lines[i].strip()

            # Check for method declaration
            method_match = re.match(r'^(static\s+)?(func\s+)(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\w+))?', line)
            if method_match:
                is_static = bool(method_match_match.group(1))
                method_name = method_match.group(3)
                params_str = method_match.group(4)
                return_type = method_match.group(5)

                # Parse parameters
                parameters = self._parse_parameters(params_str)

                # Find method body
                method_body, end_line = self._extract_method_body(lines, i)

                # Check for decorators/attributes
                decorators = []
                j = i - 1
                while j >= 0 and lines[j].strip().startswith('@'):
                    decorators.append(lines[j].strip())
                    j -= 1

                methods.append(GDScriptMethodInfo(
                    name=method_name,
                    parameters=parameters,
                    return_type=return_type,
                    is_static=is_static,
                    is_virtual='virtual' in decorators,
                    is_override='override' in decorators,
                    body=method_body,
                    start_line=i,
                    end_line=end_line,
                    decorators=decorators
                ))

                i = end_line + 1
            else:
                i += 1

        return methods

    def _parse_parameters(self, params_str: str) -> List[Tuple[str, str]]:
        """Parse method parameters."""
        if not params_str.strip():
            return []

        parameters = []
        for param in params_str.split(','):
            param = param.strip()
            if ':' in param:
                name, type_hint = param.split(':', 1)
                name = name.strip()
                type_hint = type_hint.strip()
                parameters.append((name, type_hint))
            else:
                parameters.append((param, ""))

        return parameters

    def _extract_method_body(self, lines: List[str], start_line: int) -> Tuple[str, int]:
        """Extract the body of a method."""
        if start_line >= len(lines):
            return "", start_line

        # Get indentation of the method declaration
        method_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

        body_lines = []
        i = start_line + 1

        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                body_lines.append(line)
                i += 1
                continue

            current_indent = len(line) - len(line.lstrip())
            if current_indent <= method_indent and line.strip():
                break

            body_lines.append(line)
            i += 1

        return '\n'.join(body_lines), i - 1

    def _extract_properties_and_constants(self, lines: List[str], start_line: int) -> Tuple[List[Tuple[str, str, Any]], List[Tuple[str, str, Any]]]:
        """Extract properties and constants from the lines."""
        properties = []
        constants = []
        i = 0

        while i < start_line:
            line = lines[i].strip()

            # Extract constants
            const_match = re.match(r'^const\s+(\w+)(?::\s*(\w+))?\s*=\s*(.+)', line)
            if const_match:
                name = const_match.group(1)
                type_hint = const_match.group(2) or ""
                value = const_match.group(3)
                constants.append((name, type_hint, value))

            # Extract variables
            var_match = re.match(r'^var\s+(\w+)(?::\s*(\w+))?(?:\s*=\s*(.+))?', line)
            if var_match:
                name = var_match.group(1)
                type_hint = var_match.group(2) or ""
                value = var_match.group(3)
                properties.append((name, type_hint, value))

            i += 1

        return properties, constants

    def _extract_signals(self, lines: List[str], start_line: int) -> List[Tuple[str, List[str]]]:
        """Extract signals from the lines."""
        signals = []
        i = 0

        while i < start_line:
            line = lines[i].strip()

            signal_match = re.match(r'^signal\s+(\w+)\s*\(([^)]*)\)', line)
            if signal_match:
                name = signal_match.group(1)
                params_str = signal_match.group(2)

                parameters = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        parameters.append(param)

                signals.append((name, parameters))

            i += 1

        return signals

    def _extract_inner_classes(self, lines: List[str], start_line: int) -> List['GDScriptClassInfo']:
        """Extract inner classes from the lines."""
        inner_classes = []
        i = start_line

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('class '):
                # Extract class name and extends
                class_match = re.match(r'^class\s+(\w+)(?:\s+extends\s+(\w+))?:', line)
                if class_match:
                    class_name = class_match.group(1)
                    extends = class_match.group(2)

                    # Find the end of this inner class
                    class_end = self._find_class_end(lines, i)
                    class_lines = lines[i:class_end + 1]
                    class_content = '\n'.join(class_lines)

                    # Parse inner class structure recursively
                    inner_class_info = self._parse_class_structure(class_content, class_lines)
                    inner_class_info.start_line = i
                    inner_class_info.end_line = class_end
                    inner_classes.append(inner_class_info)

                    i = class_end + 1
                else:
                    i += 1
            else:
                i += 1

        return inner_classes

    def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a class definition."""
        class_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        i = start_line + 1

        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                i += 1
                continue

            current_indent = len(line) - len(line.lstrip())
            if current_indent <= class_indent and line.strip() and not line.strip().startswith('#'):
                return i - 1

            i += 1

        return len(lines) - 1

    # Validation Methods
    async def validate_gdscript(self, file_path: str) -> GDScriptValidationResult:
        """
        Validate a GDScript file for syntax and common issues.

        Args:
            file_path: Path to the GDScript file

        Returns:
            GDScriptValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []

        try:
            # Read the file
            file_result = await self.file_tools.read_file_safe(file_path)
            if not file_result.success:
                return GDScriptValidationResult(
                    is_valid=False,
                    errors=[f"Failed to read file: {file_result.error}"],
                    warnings=[],
                    suggestions=[]
                )

            content = file_result.new_content

            # Basic syntax validation
            syntax_errors = self._validate_syntax(content)
            errors.extend(syntax_errors)

            # Structural validation
            try:
                class_info = await self.analyze_gdscript(file_path)
                structure_warnings = self._validate_structure(class_info)
                warnings.extend(structure_warnings)
            except Exception as e:
                errors.append(f"Failed to analyze structure: {str(e)}")

            # Style and best practices
            style_suggestions = self._analyze_style(content)
            suggestions.extend(style_suggestions)

            is_valid = len(errors) == 0

            logger.info(f"GDScript validation complete for {file_path}: {'VALID' if is_valid else 'INVALID'}")
            return GDScriptValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        except Exception as e:
            logger.error(f"Error validating GDScript {file_path}: {e}")
            return GDScriptValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    def _validate_syntax(self, content: str) -> List[str]:
        """Validate basic GDScript syntax."""
        errors = []

        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []

        for i, char in enumerate(content):
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if not stack:
                    errors.append(f"Unmatched closing bracket '{char}' at position {i}")
                else:
                    open_char, open_pos = stack.pop()
                    if brackets[open_char] != char:
                        errors.append(f"Mismatched brackets: '{open_char}' at {open_pos} and '{char}' at {i}")

        if stack:
            for open_char, pos in stack:
                errors.append(f"Unclosed bracket '{open_char}' at position {pos}")

        # Check for method signature issues
        method_pattern = r'func\s+\w+\s*\([^)]*\)'
        methods = re.finditer(method_pattern, content)
        for match in methods:
            method_text = match.group()
            if method_text.count('(') != method_text.count(')'):
                errors.append(f"Malformed method signature: {method_text}")

        return errors

    def _validate_structure(self, class_info: GDScriptClassInfo) -> List[str]:
        """Validate class structure and common issues."""
        warnings = []

        # Check for missing class_name
        if class_info.name is None:
            warnings.append("Consider adding a class_name for better organization")

        # Check for methods without return types
        for method in class_info.methods:
            if method.return_type is None and method.name not in ['_ready', '_process', '_physics_process', '_input']:
                warnings.append(f"Method '{method.name}' lacks return type annotation")

        # Check for unused variables (basic check)
        used_vars = set()
        for method in class_info.methods:
            # Find variable usage in method bodies
            for prop_name, _, _ in class_info.properties:
                if prop_name in method.body:
                    used_vars.add(prop_name)

        for prop_name, _, _ in class_info.properties:
            if prop_name not in used_vars and not prop_name.startswith('_'):
                warnings.append(f"Property '{prop_name}' appears to be unused")

        return warnings

    def _analyze_style(self, content: str) -> List[str]:
        """Analyze code style and suggest improvements."""
        suggestions = []

        # Check line length
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                suggestions.append(f"Line {i} is very long ({len(line)} characters). Consider breaking it up.")

        # Check for missing type hints
        func_pattern = r'func\s+(\w+)\s*\(([^)]*)\)(?!.*->)'
        matches = re.finditer(func_pattern, content)
        for match in matches:
            method_name = match.group(1)
            if method_name not in ['_ready', '_process', '_physics_process', '_input']:
                suggestions.append(f"Consider adding return type annotation to method '{method_name}'")

        # Check for magic numbers
        number_pattern = r'\b\d+\b'
        numbers = re.finditer(number_pattern, content)
        for match in numbers:
            number = match.group()
            if number not in ['0', '1', '2'] and not match.start() > 0 or content[match.start()-1] != '.':
                # This is a magic number that should probably be a constant
                line_num = content[:match.start()].count('\n') + 1
                suggestions.append(f"Consider replacing magic number {number} on line {line_num} with a named constant")

        return suggestions

    # Advanced Editing Methods
    async def refactor_method(
        self,
        file_path: str,
        method_name: str,
        new_name: Optional[str] = None,
        new_parameters: Optional[List[Tuple[str, str]]] = None,
        new_body: Optional[str] = None
    ) -> GDScriptEditResult:
        """
        Refactor an existing method with comprehensive changes.

        Args:
            file_path: Path to the GDScript file
            method_name: Name of the method to refactor
            new_name: New name for the method (optional)
            new_parameters: New parameter list (optional)
            new_body: New method body (optional)

        Returns:
            GDScriptEditResult with operation details
        """
        try:
            # Analyze the file
            class_info = await self.analyze_gdscript(file_path)

            # Find the method
            method = None
            for m in class_info.methods:
                if m.name == method_name:
                    method = m
                    break

            if not method:
                return GDScriptEditResult(
                    success=False,
                    file_path=file_path,
                    error=f"Method '{method_name}' not found"
                )

            # Build the new method signature
            params_str = self._build_parameters_string(new_parameters or method.parameters)
            return_type = f" -> {method.return_type}" if method.return_type else ""
            static_prefix = "static " if method.is_static else ""

            if new_name:
                signature = f"{static_prefix}func {new_name}({params_str}){return_type}"
            else:
                signature = f"{static_prefix}func {method_name}({params_str}){return_type}"

            # Add decorators
            decorators = '\n'.join(method.decorators) + '\n' if method.decorators else ""

            # Build the complete new method
            if new_body:
                new_method = f"{decorators}{signature}:\n{new_body}"
            else:
                new_method = f"{decorators}{signature}:\n{method.body}"

            # Use the file tools to modify the method
            if new_name and new_name != method_name:
                # If renaming, we need to remove old method and add new one
                remove_result = await self.file_tools.remove_gdscript_method(file_path, method_name)
                if not remove_result.success:
                    return GDScriptEditResult(
                        success=False,
                        file_path=file_path,
                        error=f"Failed to remove old method: {remove_result.error}"
                    )

                add_result = await self.file_tools.add_gdscript_method(file_path, new_method, "end")
                return GDScriptEditResult(
                    success=add_result.success,
                    file_path=file_path,
                    modified_methods=[method_name],
                    added_methods=[new_name],
                    error=add_result.error if not add_result.success else None
                )
            else:
                # Just modify the existing method
                return await self.file_tools.modify_gdscript_method(file_path, method_name, new_method)

        except Exception as e:
            logger.error(f"Error refactoring method: {e}")
            return GDScriptEditResult(
                success=False,
                file_path=file_path,
                error=str(e)
            )

    def _build_parameters_string(self, parameters: List[Tuple[str, str]]) -> str:
        """Build a parameter string from parameter list."""
        if not parameters:
            return ""

        params = []
        for name, type_hint in parameters:
            if type_hint:
                params.append(f"{name}: {type_hint}")
            else:
                params.append(name)

        return ", ".join(params)

    async def extract_method_to_new_file(
        self,
        source_file: str,
        method_name: str,
        new_file_path: str,
        class_name: Optional[str] = None
    ) -> GDScriptEditResult:
        """
        Extract a method to a new file and create a reference.

        Args:
            source_file: Original GDScript file
            method_name: Method to extract
            new_file_path: Path for the new file
            class_name: Class name for the new file (optional)

        Returns:
            GDScriptEditResult with operation details
        """
        try:
            # Analyze the source file
            class_info = await self.analyze_gdscript(source_file)

            # Find the method
            method = None
            for m in class_info.methods:
                if m.name == method_name:
                    method = m
                    break

            if not method:
                return GDScriptEditResult(
                    success=False,
                    file_path=source_file,
                    error=f"Method '{method_name}' not found"
                )

            # Create the new file content
            if class_name:
                file_header = f'class_name {class_name}\n\n'
            else:
                file_header = ""

            # Build the method for the new file
            static_prefix = "static " if method.is_static else ""
            params_str = self._build_parameters_string(method.parameters)
            return_type = f" -> {method.return_type}" if method.return_type else ""
            signature = f"{static_prefix}func {method_name}({params_str}){return_type}"

            decorators = '\n'.join(method.decorators) + '\n' if method.decorators else ""
            new_file_content = f"{file_header}{decorators}{signature}:\n{method.body}"

            # Write the new file
            write_result = await self.file_tools.write_file_safe(new_file_path, new_file_content)
            if not write_result.success:
                return GDScriptEditResult(
                    success=False,
                    file_path=source_file,
                    error=f"Failed to write new file: {write_result.error}"
                )

            # Remove the method from the original file
            remove_result = await self.file_tools.remove_gdscript_method(source_file, method_name)
            if not remove_result.success:
                return GDScriptEditResult(
                    success=False,
                    file_path=source_file,
                    error=f"Failed to remove method from original file: {remove_result.error}"
                )

            logger.info(f"Successfully extracted method '{method_name}' to {new_file_path}")
            return GDScriptEditResult(
                success=True,
                file_path=source_file,
                removed_methods=[method_name]
            )

        except Exception as e:
            logger.error(f"Error extracting method: {e}")
            return GDScriptEditResult(
                success=False,
                file_path=source_file,
                error=str(e)
            )


# Convenience function for direct tool access
@tool
async def analyze_gdscript_structure(file_path: str) -> GDScriptClassInfo:
    """Analyze the structure of a GDScript file.

    Args:
        file_path: Path to the GDScript file to analyze

    Returns:
        GDScriptClassInfo containing complete structural analysis
    """
    editor = GDScriptEditor()
    return await editor.analyze_gdscript(file_path)


@tool
async def validate_gdscript_syntax(file_path: str) -> GDScriptValidationResult:
    """Validate a GDScript file for syntax and structural issues.

    Args:
        file_path: Path to the GDScript file to validate

    Returns:
        GDScriptValidationResult containing validation details
    """
    editor = GDScriptEditor()
    return await editor.validate_gdscript(file_path)


@tool
async def refactor_gdscript_method(
    file_path: str,
    method_name: str,
    **kwargs
) -> GDScriptEditResult:
    """Refactor an existing GDScript method.

    Args:
        file_path: Path to the GDScript file
        method_name: Name of the method to refactor
        **kwargs: Refactoring options (new_name, new_parameters, new_body)

    Returns:
        GDScriptEditResult containing operation details
    """
    editor = GDScriptEditor()
    return await editor.refactor_method(file_path, method_name, **kwargs)


@tool
async def extract_gdscript_method(
    source_file: str,
    method_name: str,
    new_file_path: str,
    **kwargs
) -> GDScriptEditResult:
    """Extract a method to a new GDScript file.

    Args:
        source_file: Original GDScript file
        method_name: Method to extract
        new_file_path: Path for the new file
        **kwargs: Additional options (class_name)

    Returns:
        GDScriptEditResult containing operation details
    """
    editor = GDScriptEditor()
    return await editor.extract_method_to_new_file(source_file, method_name, new_file_path, **kwargs)
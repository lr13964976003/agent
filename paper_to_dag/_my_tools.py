from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import subprocess
import re

class PythonFileInput(BaseModel):
    filepath: str = Field(..., description="The path of the python file to be executed")

class PythonTool(BaseTool):
    name : str = "Python File Execute"
    description : str = "execute python file and return result"
    args_schema: Type[BaseModel] = PythonFileInput
    def _run(self, filepath: str) -> str:
        try:
            result = subprocess.run(
                ["python3", filepath],
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"execute error {e}"

    async def _arun(self, code: str) -> str:
        raise NotImplementedError

class CommandInput(BaseModel):
    command: list[str] = Field(..., description="The shell command to execute, provided as a list of strings. Example: ['ls', '/home']")

class CommandTool(BaseTool):
    name: str = "Command Executor"
    description: str = "Execute shell commands and return their output"
    args_schema: Type[BaseModel] = CommandInput

    def _run(self, command: list[str]) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout + result.stderr

        except Exception as e:
            return f"Command execution error: {e}"

    async def _arun(self, command: list[str]) -> str:
        raise NotImplementedError("Async execution not implemented yet")


class AppendFileInput(BaseModel):
    dstfilepath: str = Field(..., description="The path of file to be appended")
    srcfilepath: str = Field(..., description="The file path for appending the contents of the file to the end of another file")


class AppendFileTool(BaseTool):
    name: str = "Append file"
    description: str = "Add the content of one file to the end of another file"
    args_schema: Type[BaseModel] = AppendFileInput

    def _run(self, dstfilepath: str, srcfilepath: str) -> str:
        try:
            with open(dstfilepath, "a") as f1, open(srcfilepath, "r") as f2:
                f1.write(f2.read() + "\n")
            return f"Success add {srcfilepath} to {dstfilepath}"
        except Exception as e:
            return f"add failed: {e}"

class ReadPdfInput(BaseModel):
    filepath: str = Field(..., description="The file to be appended")

class ReadPdfTool(BaseTool):
    name: str = "Read PDF File"
    description: str = "Read the content of PDF file"
    args_schema: Type[BaseModel] = ReadPdfInput

    def _run(self, filepath: str) -> str:
        try:
            import PyPDF2
            with open(filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            return f"read pdf failed: {e}"

def parse_edges(edge_list):
    edges = []
    for pair in edge_list:
        src, dst = pair.split("->")
        edges.append([src.strip(), dst.strip()])
    return edges

def has_cycle(edges):
    graph = {}
    for src, dst in edges:
        graph.setdefault(src,[]).append(dst)

    visited = set()
    rec_stack = set()

    def dfs(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True

        rec_stack.remove(node)
        return False

    start = next(iter(graph))
    return dfs(start)

class ExtractEdgeFromDAGInput(BaseModel):
    dagpath: str = Field(..., description="The path of dag")

class ExtractEdgeFromDAGTool(BaseTool):
    name: str = "Extract Info From DAG "
    description: str = "Extract edges and nodes with only in-degree or out-degree and whether there has a cycle from DAG"
    args_schema: Type[BaseModel] = ExtractEdgeFromDAGInput
    
    def _run(self, dagpath: str) -> str:
        try:
            edge_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_]+)', re.MULTILINE)
            edges = []
            with open(dagpath, 'r', encoding='utf-8') as f:
                for line in f:
                    match = edge_pattern.match(line)
                    if match:
                        edges.append(f"{match.group(1)} -> {match.group(2)}")

            in_nodes = set()
            out_nodes = set()
            for src, dst in parse_edges(edges):
                out_nodes.add(src)
                in_nodes.add(dst)
            only_in = in_nodes - out_nodes
            only_out = out_nodes - in_nodes

            

            return {"edges": edges, "nodes with only in": only_in, "nodes with only out": only_out, "has_cycle": has_cycle(parse_edges(edges))}
        except Exception as e:
            return f"extract edge failed: {e}"


class SearchFileInput(BaseModel):
    file: str = Field(..., description="Path to the file to search")
    query: str = Field(..., description="Text to search for (case-insensitive)")
    context_lines: int = Field(2, description="Number of lines of context to show before and after each match (default: 2)")
    max_matches: int = Field(5, description="Maximum number of matches to return per page(default: 5)")
    start_idx: int = Field(0, description="From which matched content to start deplaying")


class SearchFileTool(BaseTool):
    name: str = "Search Text in File"
    description: str = "Search for a keyword or phrase in a file and return matching lines with context"
    args_schema: Type[BaseModel] = SearchFileInput

    def _run(self, file: str, query: str, context_lines: int = 2, max_matches: int = 5, start_idx: int = 0) -> str:
        try:
            with open(file, "r") as f:
                lines = f.readlines()
            
            all_matches = []
            query = query.lower()

            for i,line in enumerate(lines):
                if query in line.lower():
                    start = max(0, 1 - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    context = []
                    for j in range(start, end):
                        prefix = ">>> " if j == i else "   "
                        context.append(f"{prefix}{j+1}:{lines[j]}")
                    all_matches.append("\n".join(context))

            if not all_matches:
                return f"No matches found for {query} in {file}"

            total_matches = len(all_matches)

            end_idx = min(start_idx + max_matches, total_matches)

            matches = all_matches[start_idx:end_idx]

            summary = [
                f"Found {total_matches} matches for {query} in {file}",
                f"Showing matches {start_idx + 1}-{end_idx})",
                ""
            ]

            numbered_matches = []
            for i,_match in enumerate(matches, start=start_idx + 1):
                numbered_matches.append(f"[Match {i} of {total_matches}]\n{_match}")
            return "\n\n".join(summary + numbered_matches)

        except Exception as e:
            return f"search file failed: {e}"

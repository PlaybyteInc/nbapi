from dataclasses import dataclass
from dataclasses_json import dataclass_json
import io
from typing import Any
import nbformat
from nbclient import NotebookClient
import json
import re
import requests

PARAM_QUERY = re.compile('^[ \t]*([a-z_\-\d]+)[ \t]*=[ \t]*([^#\s]+)[ \t]*#[ \t]*@param[ \t]*(.+)', re.IGNORECASE)

DataType = str

@dataclass_json
@dataclass
class Value:
    input: str | None = None
    constant: str | None = None

    def resolve(self, input: dict[str, str]) -> str:
        if self.constant:
            return self.constant
        elif self.input:
            return input[self.input]
        else:
            return ""

@dataclass_json
@dataclass
class Artifact:
    path: str
    mimetype: str

@dataclass_json
@dataclass
class Stage:
    vars: dict[str, Value] # map of input key to colab param ident
    cell_id: str | None = None
    source: str | None = None

@dataclass_json
@dataclass
class Service:
    url: str
    input: dict[str, DataType]
    output: dict[str, Artifact]
    plan: list[Stage]

async def parse(nb_url: str) -> Service:
    source = requests.get(nb_url).text
    nb = nbformat.reads(source, nbformat.NO_CONVERT)
    plan = []
    for cell in nb.cells:
        if cell["cell_type"] != "code":
            continue
        id = _parse_cell_id(cell)
        cell_vars = _parse_cell_vars(cell)
        if id:
            plan.append(Stage(cell_vars, cell_id=id, source=None))
    return Service(nb_url, input={}, output={}, plan=plan)

async def exec(service: Service, input: dict[str, dict[str, str]]):
    source = requests.get(service.url).text
    nb = nbformat.reads(source, as_version=nbformat.NO_CONVERT)
    cells = {}
    indicies = {}
    for (index, cell) in enumerate(nb.cells):
        if cell["cell_type"] != "code":
            continue
        id = _parse_cell_id(cell)
        if id:
            cells[id] = cell
            indicies[id] = index

    client = NotebookClient(nb)
    async with client.async_setup_kernel():
        for stage in service.plan:
            if stage.cell_id:
                cell = cells[stage.cell_id]
                _insert_vars_in_cell(cell, stage.vars, input)
                await client.async_execute_cell(cell, indicies[stage.cell_id])
            if stage.source:
                source = _insert_vars_in_source(stage.source, stage.vars, input)
                await client.kc.execute(source)

def _parse_cell_vars(cell) -> dict[str, Value] | None:
    vars = {}
    source = cell["source"] if "source" in cell else ""
    for line in source.splitlines():
        match = PARAM_QUERY.match(line)
        if match == None:
            continue
        groups = match.groups() # [ident, default, options]
        vars[groups[0]] = Value(input=None, constant=groups[1])

    return vars

def _parse_cell_id(cell) -> str | None:
    if not "metadata" in cell:
        return None
    metadata = cell["metadata"]
    if not "id" in metadata:
        return None
    return metadata["id"]

def _insert_vars_in_cell(cell: dict[str, str], vars: dict[str, Value], input: dict[str, str]):
    source = cell["source"] if "source" in cell else ""
    cell["source"] = _insert_vars_in_source(source, vars, input)

def _insert_vars_in_source(source: str, vars: dict[str, Value], input: dict[str, str]):
    lines = list(_insert_vars_in_line(line, vars, input) for line in source.splitlines())
    return "\n".join(lines)

def _insert_vars_in_line(line: str, vars: dict[str, Value], input: dict[str, str]):
    if PARAM_QUERY.match(line) == None:
        return line
    parts = re.split(PARAM_QUERY, line)
    [prefix, varname, value, info, suffix] = parts
    if varname in vars:
        value = vars[varname].resolve(input)
    return f'{prefix}{varname} = {value} #@param {info}{suffix}'

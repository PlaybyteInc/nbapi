from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Any, Dict, List, Optional
import nbformat
from nbclient import NotebookClient
import re
import requests
import logging
import sys

PARAM_QUERY = re.compile('^[ \t]*([a-z_\-\d]+)[ \t]*=[ \t]*([^#\s]+)[ \t]*#[ \t]*@param[ \t]*(.+)', re.IGNORECASE)

DataType = str

@dataclass_json
@dataclass
class Value:
    input: Optional[str] = None
    constant: Optional[str] = None

    def resolve(self, input: Dict[str, str]) -> str:
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
    mimetype: Optional[str] = None

@dataclass_json
@dataclass
class Stage:
    vars: Dict[str, Value] # map of input key to colab param ident
    cell_id: Optional[str] = None
    source: Optional[str] = None

@dataclass_json
@dataclass
class Service:
    url: str
    input: Dict[str, DataType]
    output: Dict[str, Artifact]
    plan: List[Stage]

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

async def exec(service: Service, input: Dict[str, Dict[str, str]]):
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

    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    client = NotebookClient(nb)
    client.log = logger
    async with client.async_setup_kernel():
        for stage in service.plan:
            if stage.cell_id:
                cell = cells[stage.cell_id]
                _insert_vars_in_cell(cell, stage.vars, input)
                await client.async_execute_cell(cell, indicies[stage.cell_id])
            if stage.source:
                source = _insert_vars_in_source(stage.source, stage.vars, input)
                await client.kc.execute(source)

def _parse_cell_vars(cell) -> Optional[Dict[str, Value]]:
    vars = {}
    source = cell["source"] if "source" in cell else ""
    for line in source.splitlines():
        match = PARAM_QUERY.match(line)
        if match == None:
            continue
        groups = match.groups() # [ident, default, options]
        vars[groups[0]] = Value(input=None, constant=groups[1])

    return vars

def _parse_cell_id(cell) -> Optional[str]:
    if not "metadata" in cell:
        return None
    metadata = cell["metadata"]
    if not "id" in metadata:
        return None
    return metadata["id"]

def _insert_vars_in_cell(cell: Dict[str, str], vars: Dict[str, Value], input: Dict[str, str]):
    source = cell["source"] if "source" in cell else ""
    cell["source"] = _insert_vars_in_source(source, vars, input)

def _insert_vars_in_source(source: str, vars: Dict[str, Value], input: Dict[str, str]):
    lines = list(_insert_vars_in_line(line, vars, input) for line in source.splitlines())
    return "\n".join(lines)

def _insert_vars_in_line(line: str, vars: Dict[str, Value], input: Dict[str, str]):
    if PARAM_QUERY.match(line) == None:
        return line
    parts = re.split(PARAM_QUERY, line)
    [prefix, varname, value, info, suffix] = parts

    if varname in vars:
        value = vars[varname].resolve(input)
        decoded_info = decode_info(info)
        if "type" in decoded_info and decoded_info["type"] == "string":
            value = f'"{value}"'
    return f'{prefix}{varname} = {value} #@param {info}{suffix}'

# Converts the string '{type: "<type>"}' to a dict
def decode_info(info: str):
    # We can't treat info as json because the keys are not quoted
    # so we need to do some string manipulation
    if info == None or info == "":
        return None
    if info[0] == "{" and info[-1] == "}":
        info = info[1:-1]
    info_dict = {}
    for key_value in info.split(","):
        key_value = key_value.strip()
        if key_value == "":
            continue
        [key, value] = key_value.split(":")
        key = key.strip()
        value = value.strip()
        if value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        info_dict[key] = value
    return info_dict

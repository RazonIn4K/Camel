"""Type stubs for pkg_resources."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

class Distribution:
    project_name: str
    version: str
    location: str
    extras: List[str]
    activated: bool
    requires: List[str]

    def __init__(
        self,
        project_name: str,
        version: str = "",
        location: str = "",
        extras: Optional[List[str]] = None,
        requires: Optional[List[str]] = None,
    ) -> None: ...
    def get_entry_map(
        self, group: Optional[str] = None
    ) -> Dict[str, Dict[str, "EntryPoint"]]: ...
    def get_entry_info(self, group: str, name: str) -> Optional["EntryPoint"]: ...

    # Define requires method differently to avoid name clash
    def get_requires(
        self, extras: Optional[List[str]] = None
    ) -> List["Requirement"]: ...

class Requirement:
    project_name: str
    key: str
    extras: Set[str]
    specifier: str

    def __init__(self, req_string: str) -> None: ...
    @staticmethod
    def parse(req_string: str) -> "Requirement": ...

class EntryPoint:
    name: str
    module_name: str
    attrs: List[str]
    extras: List[str]
    dist: Optional[Distribution]

    def __init__(
        self,
        name: str,
        module_name: str,
        attrs: Optional[List[str]] = None,
        extras: Optional[List[str]] = None,
        dist: Optional[Distribution] = None,
    ) -> None: ...
    def load(self) -> Any: ...
    def require(self, extras: Optional[List[str]] = None) -> None: ...

class Environment:
    def __init__(self, search_path: Optional[List[str]] = None) -> None: ...
    def can_add(self, dist: Distribution) -> bool: ...
    def remove(self, dist: Distribution) -> None: ...
    def scan(self, search_path: Optional[List[str]] = None) -> None: ...

class WorkingSet:
    entries: List[str]

    def __init__(self, entries: Optional[List[str]] = None) -> None: ...
    def add_entry(self, entry: str) -> None: ...
    def find(self, req: Union[Requirement, str]) -> Optional[Distribution]: ...
    def iter_entry_points(
        self, group: str, name: Optional[str] = None
    ) -> Iterator[EntryPoint]: ...
    def run_script(self, requires: str, script_name: str) -> None: ...

def declare_namespace(packageName: str) -> None: ...
def resource_exists(
    package_or_requirement: Union[str, Requirement, Distribution], resource_name: str
) -> bool: ...
def resource_stream(
    package_or_requirement: Union[str, Requirement, Distribution], resource_name: str
) -> Any: ...
def resource_string(
    package_or_requirement: Union[str, Requirement, Distribution], resource_name: str
) -> bytes: ...
def resource_isdir(
    package_or_requirement: Union[str, Requirement, Distribution], resource_name: str
) -> bool: ...
def resource_listdir(
    package_or_requirement: Union[str, Requirement, Distribution], resource_name: str
) -> List[str]: ...
def resource_filename(
    package_or_requirement: Union[str, Requirement, Distribution], resource_name: str
) -> str: ...
def set_extraction_path(path: str) -> None: ...
def cleanup_resources(force: bool = False) -> List[str]: ...
def get_distribution(dist: Union[str, Requirement, Distribution]) -> Distribution: ...
def load_entry_point(
    dist: Union[str, Requirement, Distribution], group: str, name: str
) -> Any: ...
def get_entry_map(
    dist: Union[str, Requirement, Distribution], group: Optional[str] = None
) -> Dict[str, Dict[str, EntryPoint]]: ...
def get_entry_info(
    dist: Union[str, Requirement, Distribution], group: str, name: str
) -> Optional[EntryPoint]: ...
def iter_entry_points(
    group: str, name: Optional[str] = None
) -> Iterator[EntryPoint]: ...
def require(*requirements: Union[str, Requirement]) -> List[Distribution]: ...
def find_distributions(
    path_item: str, only: bool = False
) -> Iterator[Distribution]: ...

working_set: WorkingSet

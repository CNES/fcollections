from __future__ import annotations

import abc
import dataclasses as dc
import functools
import logging
import os
import typing as tp
import warnings
from enum import Enum, auto
from pathlib import Path

import fsspec
import pandas as pda

from ._codecs import DecodingError
from ._filenames import FileNameConvention, FileNameField

logger = logging.getLogger(__name__)


class FileListingError(Exception):
    """Raised when an error occurs during files discovery and
    interpretation."""


class ILayout:
    """Information about a multiple Tree levels.

    Given a Tree (ex a filesystem) with a structure of N homogeneous levels,
    the layout will associate each level with a FileNameConvention to extract
    useful information. This information can then be leveraged by building
    filters to speed up the tree visitation.

    For example, let's consider a set of altimetry data files, organized in
    pre-defined folders: v1/Expert/cycle_001, v1/Expert/cycle_002,
    v2/Basic/cycle_001, ...
    The first level contains information about the version, the second level
    about the subset, and the last level about the cycle number. The layout will
    declare three FileNameConvention to 'know' about the tree structure. Then,
    filters - for example subset='Expert' - can be set to select only a subpart
    of the tree, greatly improving the visitation performance.
    """

    @property
    @abc.abstractmethod
    def names(self) -> set[str]:
        """Names of the supported filters."""

    @abc.abstractmethod
    def generate(self, root: str, **fields: tp.Any) -> str:
        """Generate a path from the fields.

        Parameters
        ----------
        root
            The root path
        fields
            key/values for interpolating the conventions

        Returns
        -------
        :
            A path

        Raises
        ------
        ValueError
            In case one of the field required to generate the path is missing,
        ValueError
            In case one of the field required to generate the path has an
            improper value
        """

    @abc.abstractmethod
    def set_filters(self, **references: tp.Any):
        """Set filters used to check if a path complies with the layout.

        Parameters
        ----------
        **references
            Key/values matching at least one of the underlying conventions
        """

    @abc.abstractmethod
    def test(self, level: int, node: str) -> bool:
        """Checks if a path part matches the current filters.

        Parameters
        ----------
        node
            Path part that needs to be checked
        level
            Level of the current path part among the layout conventions

        Returns
        -------
        :
            True if the path part is selected with the current filters, False
            otherwise
        """


class Layout(ILayout):
    """Implements a ILayout with a succession of conventions.

    Parameters
    ----------
    conventions
        List of convention, with the first element matching the tree root, and
        last element the last level before the leafs

    See Also
    --------
    fcollections.core.FileNameConvention: the equivalent for the tree leafs (file
    names)
    """

    def __init__(self, conventions: list[FileNameConvention]):
        self.conventions = conventions
        self.set_filters()

    def generate(self, root: str, **fields: tp.Any) -> str:
        elements = []
        for convention in self.conventions:
            names = [f.name for f in convention.fields]
            element = convention.generate(
                **{k: v for k, v in fields.items() if k in names}
            )
            elements.append(element)

        return os.path.join(root, *elements)

    def set_filters(self, **references: tp.Any):
        filters = []
        unknown_references = set(references)
        for level, convention in enumerate(self.conventions):
            names = [f.name for f in convention.fields]

            filtered_references = {k: v for k, v in references.items() if k in names}
            unknown_references -= set(filtered_references)
            logger.debug(
                "Setting layout level %d filters to %s", level, filtered_references
            )
            filters.append(RecordFilter(convention.fields, **filtered_references))

        if len(unknown_references) > 0:
            msg = (
                "Layout has been configured with unknown references "
                f"'{unknown_references}'. They will be ignored."
            )
            logger.debug(msg)

        self.filters: list[RecordFilter] = filters

    def parse_node(self, level: int, node: str) -> tuple[tp.Any, ...]:
        """Interprets a node name.

        Parameters
        ----------
        level
            Depth in the layout. Depth in the layout is the depth of the node
            with respect to its root minus 1. There is no semantic for the root
            node, which explains this discrepency of layout-depth and tree-depth
        node
            Node name (not its full path)

        Returns
        -------
        :
            Structure information about the node
        """
        convention = self.conventions[level]
        try:
            match_object = convention.match(node)
            if len(convention.fields) == 0 and match_object is None:
                return None
            return convention.parse(match_object)
        except (DecodingError, AttributeError):
            return None

    def test_record(self, level: int, record: tuple[tp.Any, ...]) -> bool:
        """Checks if the node information matches the filters.

        The test will look for filters at the considered layout depth, and apply
        them on the record.

        Parameters
        ----------
        level
            Depth in the layout. Depth in the layout is the depth of the node
            with respect to its root minus 1. There is no semantic for the root
            node, which explains this discrepency of layout-depth and tree-depth
        record
            Interpreted node informations

        Returns
        -------
        :
            True if the node matches the filters, false otherwise
        """
        return self.filters[level].test(record)

    @property
    def names(self) -> set[str]:
        return set(
            functools.reduce(
                lambda x, y: x + y,
                map(
                    lambda convention: [f.name for f in convention.fields],
                    self.conventions,
                ),
            )
        )


class INode(abc.ABC):
    """Representation of a file system path.

    Parameters
    ----------
    name
        Name of the node. Not to be confused with the full path that should be
        contained in the info parameter
    info
        Additional information. ``name`` - representing the full path - is
        expected to be in this parameter. Other information will depend on the
        ``fsspec`` implementations
    level
        Nesting level of the current node with respect to the tree root
    """

    def __init__(self, name: str, info: dict[str, tp.Any], level: int):
        self.name = name
        self.info = info
        self.level = level

    @abc.abstractmethod
    def accept(self, visitor: LayoutVisitor) -> VisitResult:
        """Accept a visitor.

        This method should trigger operations in the visitor. The visitor
        computes the desired result, and the node is responsible for emitting
        said-result to the walk operation.

        Returns
        -------
        :
            The visit result

        See Also
        --------
        walk
            Walk operation handling the tree traversal
        """

    @abc.abstractmethod
    def children(self) -> tp.Iterator[INode]:
        """List child nodes.

        Returns
        -------
        :
            The child nodes, either files or folders
        """

    @abc.abstractmethod
    def clear(self):
        """Clear child nodes."""


class FileNode(INode):
    """File node of a file system tree."""

    def accept(self, visitor: LayoutVisitor) -> VisitResult:
        return visitor.visit_file(self)

    def children(self) -> tp.Iterator[INode]:
        """List child nodes.

        Returns
        -------
        :
            An empty list (files have no children)
        """
        return []

    def clear(self):
        pass


class DirNode(INode):
    """Directory node of a file system tree.

    Parameters
    ----------
    name
        Name of the node. Not to be confused with the full path that should be
        contained in the info parameter
    info
        Additional information. The entry ``name`` - representing the full path
        - is expected to be in this parameter. Other information will depend on
        the ``fsspec`` implementation
    level
        Nesting level of the current node with respect to the tree root
    fs
        File system hosting the node. Useful to list the children
    follow_symlinks
        If False, symbolic links will be marked as file nodes instead of
        directory nodes, and will not be explored
    """

    def __init__(
        self,
        name,
        info,
        fs: fsspec.AbstractFileSystem,
        level: int,
        follow_symlinks: bool = False,
    ):
        super().__init__(name, info, level)
        self.fs = fs
        self.follow_symlinks = follow_symlinks
        self._children: list[INode] | None = None

    def accept(self, visitor: LayoutVisitor) -> VisitResult:
        return visitor.visit_dir(self)

    def children(self) -> tp.Iterable[INode]:
        # Cache children computation to avoid expensive relisting and ensure
        # that one path on the filesystem will be represented by the same node
        if self._children is None:
            self._children = list(self._compute_children())
        return self._children

    def clear(self):
        self._children = None

    def _compute_children(self) -> tp.Iterator[INode]:
        # return list of FileNode or DirNode instances
        # Block of code extracted from fsspec and simplified (no topbottom
        # option)
        path = self.fs._strip_protocol(self.info["name"])

        try:
            listing = self.fs.ls(path, detail=True)
        except (FileNotFoundError, OSError):
            return

        for info in listing:
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            pathname = info["name"].rstrip("/")
            name = pathname.rsplit("/", 1)[-1]
            if info["type"] == "directory" and pathname != path:
                # do not include "self" path
                yield DirNode(name, info, self.fs, self.level + 1, self.follow_symlinks)
            elif self.follow_symlinks and "islink" in info and info["islink"]:
                info["name"] = Path(info["name"]).resolve().as_posix()
                yield DirNode(name, info, self.fs, self.level + 1, self.follow_symlinks)
            elif pathname == path:
                # file-like with same name as give path
                # consequence of virtual directories in cloud object stores
                yield FileNode("", info, self.level + 1)
            else:
                yield FileNode(name, info, self.level + 1)


@dc.dataclass(frozen=True)
class VisitResult:
    """Result of a visit.

    The result type is defined by the :class:`IVisitor` implementations.

    Additional information related to semantic definition contained in layouts
    (:class:`Layout`) is given for further advancement of the visitors.

    Tree traversal can also use exploration hints given by the visitors
    decide if the current branch should be explored.

    See Also
    --------
    walk
        Handle tree traversal
    """

    explore_next: bool
    """True if we should continue to explore the current branch."""
    payload: tp.Any | None = None
    """Post processing result of a node by the visitor."""
    surviving_layouts: list[Layout] = dc.field(default_factory=list)
    """:class:`LayoutVisitor` only, used to know which semantic is still valid
    for the current branch."""


class IVisitor(abc.ABC):
    """Visitor processing an :class:`INode`.

    Visitors interpret a node and return information from it. It is up
    to the implementation to define which information it can get from
    the node. Some implementations will only return the node path, other
    will try to interpret it using semantics' definitions.

    An important characteristic of the visitor is its ability to advance
    from a previous visit result. This gives flexibility to implement
    specific states during the tree traversal.

    Additionnal metadata about the visit are also returned by the
    visitor. This information should be used for tree traversal and
    visitor advancement only, and not returned by the walk operation.
    """

    @abc.abstractmethod
    def visit_dir(self, dir_node: DirNode) -> VisitResult:
        """Visits a directory node.

        Parameters
        ----------
        dir_node
            The directory node to visit

        Returns
        -------
        :
            Node information and visit metadata.
        """

    @abc.abstractmethod
    def visit_file(self, file_node: DirNode) -> VisitResult:
        """Visits a file node.

        Parameters
        ----------
        file_node
            The file node to visit

        Returns
        -------
        :
            Node information and visit metadata.
        """

    @abc.abstractmethod
    def advance(self, result: VisitResult) -> IVisitor:
        """Advance the visitor.

        The advancement can either return a reference or a copy of the visitor.
        If a per-branch state is needed, it is advised to return a copy.

        Parameters
        ----------
        result
            Previous result of a visit. Originally intended to be the visit
            result of the parent node.

        Returns
        -------
        :
            The current visitor or a copy with a modified state
        """


class StandardVisitor(IVisitor):
    """Visitor for producing the equivalent of
    :meth:`fsspec.spec.AbstractFileSystem.walk`.

    The useful information is a tuple (root, dirs, files) that mimics
    the standard output of a walk operation.

    No additionnal metadata related to the visit itself is returned.
    """

    def visit_dir(self, dir_node: DirNode) -> VisitResult:
        dirs = []
        files = []
        for x in dir_node.children():
            if isinstance(x, DirNode):
                dirs.append(x.name)
            else:
                files.append(x.name)
        return VisitResult(True, (dir_node.info["name"], dirs, files))

    def visit_file(self, file_node: DirNode) -> VisitResult:
        # No payload in visit_file
        return VisitResult(False)

    def advance(self, result: VisitResult) -> StandardVisitor:
        # Visitor should advance without copy, duplication or state alteration
        return self


class VisitError(Exception):
    """Raised by the visitor during node visit."""


class LayoutMismatchError(VisitError):
    """Raised if all layouts do not match the actual file system structure."""


class LayoutMismatchHandling(Enum):
    """Possibilities when a folder of file node does not match any layout."""

    RAISE = auto()
    """Raise an exception."""
    WARN = auto()
    """Warn the user."""
    IGNORE = auto()
    """Ignore the mismatch."""


class LayoutVisitor(IVisitor):
    """Visitor with node interpretation and branch exploration hints.

    The layouts will try to interpret a node and get a record of structured
    information. Layouts also include filters that are applied to give a hint
    about tree exploration: if all layouts exclude the current node, exploration
    should not continue.

    Parameters
    ----------
    layouts
        Semantic definitions for interpreting and testing node meanings
    stat_fields
        List of node metadata to add to the record
    on_mismatch_directory
        Behavior on mismatch for directories
    on_mismatch_file
        Behavior on mismatch for files
    """

    def __init__(
        self,
        layouts: list[Layout],
        stat_fields: tp.Iterable[str] = tuple(),
        on_mismatch_directory: LayoutMismatchHandling = LayoutMismatchHandling.RAISE,
        on_mismatch_file: LayoutMismatchHandling = LayoutMismatchHandling.RAISE,
    ):
        self.layouts = layouts
        self.stat_fields = list(stat_fields)
        self.on_mismatch_directory = on_mismatch_directory
        self.on_mismatch_file = on_mismatch_file
        if "name" not in self.stat_fields:
            self.stat_fields.insert(0, "name")

    def visit_dir(self, dir_node: DirNode) -> VisitResult:
        """Visits a directory node.

        The directory node path is parsed into a structured node. If none of the
        layouts is able to parse the node, it means we are in uncharted
        territory: tree traversal hint in the visit result will state we should
        not continue exploring.

        In addition, layout filters are applied on the node information. If all
        layouts exclude the node, it means no node of interest are in this
        branch: we want to terminate the current branch exploration as soon as
        possible to speed up the walk operation.

        Multiple layouts means multiple semantics are possible. This is the case
        in a heterogeneous folder. When exploring a branch, some layouts may not
        match the branch semantic. These are pruned as soon as possible, but
        only for the current branch.

        Warns
        -----
        UserWarning
            In case the dir_node does not match any configured layout and
            ``on_mismatch`` is set to ``WARN``

        Raises
        ------
        LayoutMismatchError
            In case the dir_node does not match any configured layout and
            ``on_mismatch`` is set to ``RAISE``

        Returns
        -------
        :
            Node information and visit metadata. The visit metadata includes a
            tree traversal hint for further exploration, and the surviving
            layouts that match the current branch
        """
        logger.debug("Visiting folder %s", dir_node.info["name"])
        if dir_node.level == 0:
            # No parsing nor filtering for the root node
            return VisitResult(True, None, self.layouts)

        layouts_for_children: list[Layout] = []
        record = None
        return_payload = False
        for layout in self.layouts:
            # Prune non matching layouts for this directory. We need to test all
            # layouts to eliminate non matching layouts as early as possible in
            # a given branch
            result = layout.parse_node(dir_node.level - 1, dir_node.name)
            if result is not None:
                layouts_for_children.append(layout)

            # In case the layout leaf is a folder, the exploration must be
            # stopped even though the actual DirNode has children. This is
            # because the layout won't be able to go deeper. This case arises
            # when we want to extract information at a given level in the file
            # system hierarchy.
            if result is not None and dir_node.level == len(layout.conventions):
                return_payload = True

            if record is None:
                # Do not override a valid record with a None
                record = result

        if not layouts_for_children:
            return self._on_mismatch(dir_node, self.on_mismatch_directory)

        if all(
            [
                not layout.test_record(dir_node.level - 1, record)
                for layout in layouts_for_children
            ]
        ):
            # This folder does not match the filtering criteria. It is ignored
            logger.debug(
                "Folder %s filtered out, branch exploration stopped",
                dir_node.info["name"],
            )
            return VisitResult(False)

        # Return payload for dir nodes only if dir nodes are the last convention
        # defined in the layouts.
        if return_payload:
            return VisitResult(False, record, [])
        else:
            return VisitResult(True, None, layouts_for_children)

    def visit_file(self, file_node: FileNode) -> VisitResult:
        """Visits a file node.

        The file node is interpreted to generate a record of structured
        information. The content of this record depends on the layouts
        definition. If the interpretation fails, the visit result will not
        include any information about the node.

        Layout filters are also applied to the node record. If all layouts
        exclude the node, the visit result will not include any information
        about the node.

        Raises
        ------
        KeyError
            If the requested stats_fields key are unknown for the given fsspec
            implementation

        Returns
        -------
        :
            Node information and visit metadata. For file node, no further
            exploration should be needed. In this case, surviving layouts are
            not relevant and will not be included in the visit result.
        """
        logger.debug("Visiting file %s", file_node.info["name"])
        # Advance/prune layouts for files
        for layout in self.layouts:
            record = layout.parse_node(file_node.level - 1, file_node.name)
            if record is not None and layout.test_record(file_node.level - 1, record):
                # Files are leaf, no need to continue exploration
                return VisitResult(
                    False, (*record, *[file_node.info[x] for x in self.stat_fields])
                )
            elif record is not None:
                # Leaf node should be identical for all layouts. Do not bother
                # testing all layouts if the record has already been filtered
                # out
                return VisitResult(False)
        return self._on_mismatch(file_node, self.on_mismatch_file)

    def advance(self, result: VisitResult) -> LayoutVisitor:
        return LayoutVisitor(
            result.surviving_layouts,
            self.stat_fields,
            self.on_mismatch_directory,
            self.on_mismatch_file,
        )

    def _on_mismatch(
        self, node: INode, on_mismatch: LayoutMismatchHandling
    ) -> VisitResult:
        # Outlier
        if on_mismatch == LayoutMismatchHandling.IGNORE:
            # If nobody matches, we do not want to explore further
            logger.debug(
                "Node %s does not match any layout, branch exploration stopped.",
                node.info["name"],
            )
            return VisitResult(False)

        msg = f"Node {node.info['name']} does not match any layout."
        if on_mismatch == LayoutMismatchHandling.WARN:
            warnings.warn(msg)
            return VisitResult(False)
        else:
            raise LayoutMismatchError(msg)


class NoLayoutVisitor(IVisitor):
    """Visitor with file node interpretation only.

    The given convention will interpret the file nodes, the folders are not
    interpreted.

    Parameters
    ----------
    convention
        Semantic definitions for interpreting a file node
    record
        Tester for the file node information
    stat_fields
        List of node metadata to add to the record
    """

    def __init__(
        self,
        convention: FileNameConvention,
        record_filter: RecordFilter,
        stat_fields: tp.Iterable[str] = tuple(),
    ):
        self.convention = convention
        self.record_filter = record_filter
        self.stat_fields = list(stat_fields)
        if "name" not in self.stat_fields:
            self.stat_fields.insert(0, "name")

    def visit_dir(self, dir_node: DirNode) -> VisitResult:
        """Visits a directory node.

        Transparent visit of a directory node. The visit will not return any
        information about the node. The metadata will always hint at continuing
        the branch exploration.

        Parameters
        ----------
        dir_node
            The directory node to visit

        Returns
        -------
        :
            Node information and visit metadata.
        """
        return VisitResult(True)

    def visit_file(self, file_node: DirNode) -> VisitResult:
        logger.debug("Visiting file %s", file_node.info["name"])
        # Advance/prune layouts for files
        try:
            record = self.convention.parse(self.convention.match(file_node.name))
        except (DecodingError, AttributeError):
            return VisitResult(False)

        if self.record_filter.test(record):
            # Files are leaf, no need to continue exploration
            return VisitResult(
                False, (*record, *[file_node.info[x] for x in self.stat_fields])
            )
        return VisitResult(False)

    def advance(self, result: VisitResult) -> IVisitor:
        return self


def walk(node: INode, visitor: IVisitor) -> tp.Iterator[tp.Any]:
    """Recursive walk of a file system tree.

    This is a reimplementation of the similar :func:`os.walk` and
    :meth:`fsspec.spec.AbstractFileSystem.walk`. The motivation for the
    reimplementation is that we need to inject some complex logic (node parsing
    and branch exploration) during the tree traversal.

    Parameters
    ----------
    node
        File or folder node representing a path on the filesystem
    visitor
        Visitor that will process the note and produce some results

    Raises
    ------
    VisitError
        Raised by the visitor to signal something went wrong during a node
        visit

    Yields
    ------
    :
        The results of all visits in the tree. The result type will depend on
        the visitor implementation

    See Also
    --------
    StandardVisitor
        Visitor returning (root, dirs, files) tuples similar to a
        conventionnal walk
    LayoutVisitor
        Visitor that can interpret the node paths and return structured
        information
    """
    result = node.accept(visitor)
    if result.payload is not None:
        yield result.payload
    if not result.explore_next:
        return

    for child in node.children():
        logger.debug("child %s", child.name)
        yield from walk(child, visitor.advance(result))


class RecordFilter:
    """Utility class for filtering values.

    Attributes
    ----------
    fields: List[FileNameField]
        the fields to filter
    **references:
        the values of fields used for selection
    """

    def __init__(self, fields: list[FileNameField], **references):
        self.fields = fields
        self.references = references

        fields_names = [f.name for f in fields]
        filter_keys = list(references.keys())

        # The FileNameParser will return a record that is ordered the same as
        # its fields. These fields are passed in the same order in this
        # RecordFilter, which is why we can assume that the fields and record
        # are in matching order here
        try:
            self.index_in_record = [fields_names.index(key) for key in filter_keys]
        except ValueError as exc:
            unknown_keys = set(filter_keys) - set(fields_names)
            raise FileListingError(
                f"Tried to build filter on file name fields using unknown keys: '{unknown_keys}'"
            ) from exc

        self._sanitize_references()

    def test(self, record):
        """Test if a record is filtered.

        Parameters
        ----------
        record:
            record to filter

        Returns
        -------
        boolean
            true if the record is filtered
        """
        return all(
            self.fields[index].test(reference, record[index])
            for reference, index in zip(self.references.values(), self.index_in_record)
        )

    def _sanitize_references(self):
        for (key, reference), index in zip(
            self.references.items(), self.index_in_record
        ):
            self.references[key] = self.fields[index].sanitize(reference)


class FileSystemMetadataCollector:
    """Filtered discovery and aggregation of filesystem metadata.

    Notes
    -----

    - The aggregation has yet to be implemented.
    - Only files' metadata can be collected in the current implementation

    Parameters
    ----------
    path
        path of directory containing files
    layouts
        Succession of conventions describing how to interpret the folder and
        file nodes
    root_node
        Root node representing an explorable tree. Usually represent the parent
        directory on a file system
    """

    def __init__(self, layouts: list[Layout], root_node: INode):
        self.layouts = layouts
        self.root_node = root_node

    def discover(
        self,
        predicates: tuple[tp.Callable[[tuple[tp.Any, ...]], bool], ...] = (),
        stat_fields: tuple[str] = (),
        enable_layouts: bool = True,
        **filters,
    ) -> pda.DataFrame:
        """
        Parameters
        ----------
        predicates
            Complex predicates for filtering a file's record
        stat_fields
            Name of the file metadata fields that should be returned in the
            record. The info that can be retrieved is dependent on the file
            system implementation. Check the filesystem ``ls`` method to get the
            available stat fields
        enable_layouts
            Set to True to use the layouts for directory names parsing. This
            will speed up the listing, but may raise an error if some directory
            do not match the declared layouts. Set to False to scan the entire
            directory and parse the files only
        **filters
            filters for files/folde selection over the fields declared in the
            layouts. Each field can accept a different filter value depending on
            the underlying FileNameField subclass

        Yields
        ------
        :
            The record matching the files

        Raises
        ------
        KeyError
            In case some of the requested stat_fields are not available for the
            current file system
        LayoutMismatchError
            In case ``enable_layouts`` is True and a mismatch between the
            layouts and the actual files is detected
        """
        for layout in self.layouts:
            # TODO: We should also be able to give the predicates here -> need
            # to modify the Layout interface
            layout.set_filters(**filters)

        self.root_node.clear()
        if enable_layouts:
            logger.debug("Using layouts to speed up listing")
            visitor = LayoutVisitor(self.layouts, stat_fields)
            records = walk(self.root_node, visitor)
        else:
            logger.debug("Full scan (not using layouts)")
            layout = self.layouts[-1]
            visitor = NoLayoutVisitor(
                layout.conventions[-1], layout.filters[-1], stat_fields
            )
            records = walk(self.root_node, visitor)

        for predicate in predicates:
            records = filter(predicate, records)
        yield from records

    def to_dataframe(
        self,
        predicates: tuple[tp.Callable[[tuple[tp.Any, ...]], bool], ...] = (),
        stat_fields: tuple[str] = (),
        enable_layouts: bool = True,
        **filters,
    ) -> pda.DataFrame:
        """
        Parameters
        ----------
        predicates
            Complex predicates for filtering a file or folder's record
        stat_fields
            Name of the file or folder metadata fields that should be returned
            in the record. The info that can be retrieved is dependent on the
            file system implementation. Check the filesystem ``ls`` method to
            get the available stat fields
        enable_layouts
            Set to True to use the layouts for directory names parsing. This
            will speed up the listing, but may raise an error if some directory
            do not match the declared layouts. Set to False to scan the entire
            directory and parse the files only
        **filters
            filters for files/folders selection over the fields declared in the
            file name convention and layout (optional). Each field can accept a
            different filter value depending on the underlying FileNameField
            subclass

        Yields
        ------
        :
            A pandas's dataframe containing all selected filenames + a column
            per field requested

        Raises
        ------
        KeyError
            In case some of the requested stat_fields are not available for the
            current file system
        VisitError
            In case ``enable_layouts`` is True and a mismatch between the
            layouts and the actual files is detected
        """
        file_convention = self.layouts[0].conventions[-1]
        return pda.DataFrame(
            self.discover(predicates, stat_fields, enable_layouts, **filters),
            columns=[f.name for f in file_convention.fields]
            + ["filename"]
            + list(stat_fields),
        )

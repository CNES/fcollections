"""This module provides generic classes for interacting with a file system."""

from __future__ import annotations

import abc
import dataclasses as dc
import inspect
import logging
import textwrap
import typing as tp
import warnings
from abc import ABCMeta

import docstring_parser as dcs
import pandas as pda
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from ._filenames import FileNameConvention
from ._listing import DirNode, FileSystemMetadataCollector, Layout
from ._metadata import GroupMetadata
from ._readers import IFilesReader

if tp.TYPE_CHECKING:  # pragma: no cover
    import dask.bag.core
    import xarray as xr_t

logger = logging.getLogger(__name__)


class NotExistingPathError(Exception):
    pass


class FilesDatabaseMeta(ABCMeta):
    # We need to inherit from ABCMeta because mixins might ABCMeta as a
    # metaclass. To respect the metaclass hierarchy, we directly derive from
    # ABCMeta

    def __new__(cls, clsname, bases, attrs):

        # Create the class first. We need the inheritance mecanism to get the
        # 'reader' and 'parser' class attributes that may have been defined in
        # super-classes but not in the 'clsname' class being constructed
        _create_method(attrs, "query", "_query")
        _create_method(attrs, "list_files", "_files")
        _create_method(attrs, "variables_info", "_variables_info")
        _create_method(attrs, "map", "_map")
        new_class = super().__new__(cls, clsname, bases, attrs)

        if clsname == "FilesDatabase":
            # Do not try and update the signature and docstring of our methods
            # if the BaseClass FilesDatabase has not been created already. We
            # need this BaseClass to access the default listing parameters
            return new_class

        parameters = _extract_parameters(new_class)
        setattr(new_class, "reading_parameters", parameters["reader"][1])
        setattr(
            new_class,
            "listing_parameters",
            parameters["convention"][1] | parameters["predicates"][1],
        )

        method_parameters = _combine_parameters(new_class, parameters)

        _patch_method(
            new_class,
            "query",
            "_query",
            new_class._query.__doc__,
            *method_parameters["query"],
        )
        _patch_method(
            new_class, "map", "_map", new_class._map.__doc__, *method_parameters["map"]
        )
        _patch_method(
            new_class,
            "list_files",
            "_files",
            new_class._files.__doc__,
            *method_parameters["list_files"],
        )
        _patch_method(
            new_class,
            "variables_info",
            "_variables_info",
            new_class._variables_info.__doc__,
            *method_parameters["variables_info"],
        )

        return new_class


def _extract_parameters(
    new_class: tp.Type[FilesDatabase],
) -> dict[str, tuple[dict[str, dcs.DocstringParam], dict[str, inspect.Parameter]]]:
    parameters = {}
    parameters["reader"] = _reading_parameters(new_class.reader)
    parameters["convention"] = _convention_parameters(
        new_class.layouts[0].conventions[-1]
    )
    parameters["predicates"] = _predicates_parameters(new_class.predicate_classes)
    return parameters


def _combine_parameters(
    new_class: tp.Type[FilesDatabase],
    parameters: dict[
        str, tuple[dict[str, dcs.DocstringParam], dict[str, inspect.Parameter]]
    ],
) -> dict[str, tuple[dict[str, dcs.DocstringParam], dict[str, inspect.Parameter]]]:
    out = {}

    # Mix file name convention parameters, reading parameters and predicates
    # parameters
    query_docstring = list(
        (
            parameters["reader"][0]
            | parameters["convention"][0]
            | parameters["predicates"][0]
        ).values()
    )
    query_signature = list(
        (
            parameters["reader"][1]
            | parameters["convention"][1]
            | parameters["predicates"][1]
        ).values()
    )
    out["query"] = (query_docstring, query_signature)
    out["map"] = (query_docstring, query_signature)

    # Mix (base class) listing files parameters, file name convention
    # parameters and predicates parameters
    # self is included in the listing_signature_parameters
    files_docstring = list(
        (parameters["convention"][0] | parameters["predicates"][0]).values()
    )
    files_signature = list(
        (parameters["convention"][1] | parameters["predicates"][1]).values()
    )
    out["list_files"] = (files_docstring, files_signature)

    # Filter convention parameters. Only the fields matching the unmixer
    # subsetting keys will be accepted
    def _is_subset_key(
        item: tuple[str, dcs.DocstringParam | inspect.Parameter],
    ) -> bool:
        return (
            new_class.unmixer is not None
            and item[0] in new_class.unmixer.partition_keys
        )

    info_docstring = list(
        map(lambda x: x[1], filter(_is_subset_key, parameters["convention"][0].items()))
    )
    info_signature = list(
        map(lambda x: x[1], filter(_is_subset_key, parameters["convention"][1].items()))
    )
    out["variables_info"] = (info_docstring, info_signature)
    return out


def _reading_parameters(
    reader: IFilesReader,
) -> tuple[dict[str, dcs.DocstringParam], dict[str, inspect.Parameter]]:
    if reader is None:
        return {}, {}

    reading_docstring_parameters = {
        p.arg_name: p
        for p in dcs.parse(reader.read.__doc__).params
        if p.arg_name not in ["files", "fs", "preprocess"]
    }

    reading_signature_parameters = {
        k: p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for k, p in inspect.signature(reader.read).parameters.items()
        if k
        not in [
            "files",
            "fs",
            # Xarray readers usually declares a preprocess arguments for their
            # subclasses. This is not interesting for the high level user and
            # is removed from the interface. Ex. OpenMfDataset and
            # GeoOpenMfDataset
            "preprocess",
            # Remove kwargs from the final methods. kwargs is not precise enough
            # for high level users
            "kwargs",
        ]
    }

    return reading_docstring_parameters, reading_signature_parameters


def _convention_parameters(
    parser: FileNameConvention,
) -> tuple[dict[str, dcs.DocstringParam], dict[str, inspect.Parameter]]:
    fields = parser.fields
    convention_docstring_parameters = {
        field.name: dcs.DocstringParam(
            ["param", field.name],
            textwrap.fill(field.description),
            field.name,
            # Docstrings in the project do not repeat the typing in the
            # Parameters section. We set None to comply with this implicit
            # convention
            None,
            False,
            None,
        )
        for field in fields
    }

    convention_signature_parameters = {
        f.name: inspect.Parameter(
            f.name,
            default=inspect.Parameter.empty,
            annotation=f.type,
            kind=inspect.Parameter.KEYWORD_ONLY,
        )
        for f in fields
    }

    return convention_docstring_parameters, convention_signature_parameters


def _predicates_parameters(
    predicate_classes: list[IPredicate] | None,
) -> tuple[dict[str, dcs.DocstringParam], dict[str, inspect.Parameter]]:
    if predicate_classes is None:
        return {}, {}

    docstring_parameters, signature_parameters = {}, {}
    for predicate_builder in predicate_classes:
        docstring_parameters |= {
            p.arg_name: p
            for p in dcs.parse(predicate_builder.__init__.__doc__).params
            if p not in ["self", "indexes"]
        }

        signature_parameters |= {
            k: p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            for k, p in inspect.signature(predicate_builder.__init__).parameters.items()
            if k not in ["self", "indexes"]
        }

    return docstring_parameters, signature_parameters


def _create_method(attrs: dict[str, tp.Any], name: str, internal_name: str):
    # Default override of the dynamically parametrized functions. This
    # will allow each class to define their unique query method with
    # matchin __doc__ attribute that we can override without side
    # effects from one class to another
    def wrapped(self, *args, **kwargs):
        return getattr(self, internal_name)(*args, **kwargs)

    attrs[name] = wrapped
    attrs[name].__name__ = name


def _patch_method(
    cls: tp.Type[FilesDatabase],
    name: str,
    internal_name: str,
    docstring_template: str,
    docstring_parameters: list[dcs.DocstringParam],
    signature_parameters: list[inspect.Parameter],
):

    docstring_query = dcs.parse(docstring_template)
    docstring_query.meta.extend(docstring_parameters)
    # In case no param already in the doc, rest style is used by default. Ensure
    # numpydoc style is used instead
    docstring_query.style = dcs.DocstringStyle.NUMPYDOC
    doc = dcs.compose(docstring_query)

    method_parameters = [
        p
        for name, p in inspect.signature(getattr(cls, internal_name)).parameters.items()
        if name != "kwargs"
    ]
    method_parameters += signature_parameters
    signature = inspect.Signature(method_parameters)

    getattr(cls, name).__doc__ = doc
    getattr(cls, name).__signature__ = signature


class FilesDatabase(metaclass=FilesDatabaseMeta):
    """Abstract database mapping.

    Parameters
    ----------
    path
        path to a directory containing NetCDF files
    fs
        File system hosting the files. Can be used to access local or remote
        (S3, FTP, ...) file systems. Underlying readers may not be compatible
        with all file systems implementations
    enable_layouts
        Set to True to use the layouts for directory names parsing. This will
        speed up the listing, but may raise an error if some directory does not
        match the pre-configured layouts. Set to False to scan the entire
        directory and parse the files only
    follow_symlinks
        If False, symbolic links will be marked as file nodes instead of
        directory nodes, and will not be explored

    Attributes
    ----------
    discoverer
        File discoverer. Walks in a folder (can be on a remote file system),
        parses the listed files and filters them.
    """

    layouts: list[Layout] | None = None
    """Semantic describing how the files are organized.

    Useful to extract
    information and have an efficient file system scanning. The pre-configured
    layouts can mismatch the current files organization, in which case the user
    can build its own or set ``enable_layouts`` to False.
    """
    reader: IFilesReader | None = None
    """Files reader."""
    unmixer: SubsetsUnmixer | None = None
    """Specify how to interpret the file metadata table to unmix subsets."""
    deduplicator: Deduplicator | None = None
    """Deduplicate the file metadata table of a unique subset (after
    unmixing)."""
    sort_keys: list[str] | str | None = None
    """Keys that specifies the fields used to sort the records extracted from
    the filenames.

    Useful to order the files prior to reading them.
    """
    metadata_injection: dict[str, tuple[str, ...]] | None = None
    """Configures how metadata from the files listing can be injected in a
    dataset returned from the read.

    The keys is the columns of the file metadata table, the value is a
    tuple of dimensions for insertion.
    """
    predicate_classes: list[type[IPredicate]] | None = None
    """List of predicates that are built at each query.

    The predicates intercepts the input parameters to build a custom
    record predicate. Usually, it is a complex test involving auxiliary
    data, such as ground track footprints or half_orbit/periods tables.
    """

    def __init__(
        self,
        path: str,
        fs: AbstractFileSystem = LocalFileSystem(),
        enable_layouts: bool = True,
        follow_symlinks: bool = False,
    ):
        self.path = path
        self.fs = fs

        root_node = DirNode(
            self.path, {"name": self.path}, self.fs, 0, follow_symlinks=follow_symlinks
        )

        self.discoverer = FileSystemMetadataCollector(self.layouts, root_node)
        self.enable_layouts = enable_layouts

        def raise_if_unknown_keys(
            obj: Deduplicator | SubsetsUnmixer | dict[str, tuple[str, ...]] | None,
            name: str,
        ):
            if obj is None:
                return

            fields_names = set(map(lambda f: f.name, self.parser.fields))
            try:
                unknown = set(obj.keys()) - fields_names
            except TypeError:
                unknown = obj.keys - fields_names

            if len(unknown) > 0:
                raise ValueError(f"{name} contains unknown fields: {unknown}")

        raise_if_unknown_keys(self.deduplicator, "Deduplicator")
        raise_if_unknown_keys(self.unmixer, "Subsets Unmixer")
        raise_if_unknown_keys(self.metadata_injection, "Metadata Injection")

        if not self.fs.exists(path):
            raise NotExistingPathError(
                f"The path {path} doesn't exist in the file system."
            )

    @property
    def parser(self) -> FileNameConvention:
        """Amongst all name conventions, get the one managing the files.

        Returns
        -------
        :
            Files name parser
        """
        return self.layouts[0].conventions[-1]

    def _files(
        self,
        sort: bool = False,
        deduplicate: bool = False,
        unmix: bool = False,
        predicates: tp.Iterable[IPredicate] = (),
        stat_fields: tuple[str] = (),
        **kwargs,
    ) -> pda.DataFrame:
        """List the files matching the given criteria.

        Parameters
        ----------
        sort
            Sort the results using the sort_keys attribute if this class
        deduplicate
            In case the class deduplicator is defined, the results are analyzed to
            search for duplicates according to a set of unique keys. In case
            duplicates are found, deduplication is run along a set of defined
            columns where duplicates are expected to occur
        unmix
            Multiple subsets may be mixed in the files metadata table. Use this
            argument to separate the subsets. An auto pick will also be performed
            according to the SubsetsUnmixer instance of this class. In case the
            auto pick cannot get a unique subset, an error is raised
            deduplication operation is done and if there are still duplicates,
            an error is raised
        predicates
            Additional complex filters to run on the record parsed by the
            filename. ex. ``lambda record: record[1] in [1, 4, 5]``. Predicates
            are knowledgeable about the record contents and the file name
            convention
        stat_fields
            File system information that can be retrieved from the fsspec
            underlying implementation. For example, 'size' or 'created' are
            valid for a local file system

        Raises
        ------
        ValueError
            In case unmix is True, an error is raised if one unique and
            homogeneous subset cannot be extracted from the files metadata table
        LayoutMismatchError
            In case ``enable_layouts`` is True and a mismatch between the
            layouts and the actual files is detected

        Returns
        -------
            A panda DataFrame containing the files metadata table. It contains
            at least a 'filename' column and the fields defined in the filename
            convention. In case stat_fields is not empty, the additionnal
            columns will also be displayed in the files metadata table
        """
        # This docstring will be superseded by the metaclass
        bad_kwargs = [k for k in kwargs if k not in self.listing_parameters]
        if bad_kwargs != []:
            raise ValueError(
                f"list_files() got unexpected keyword argument(s): {bad_kwargs}"
            )

        # Auto-build declared predicates. Parameters used by the predicates are
        # expected to be independant of the other parameters from the file name
        # convention
        predicates = list(predicates)
        if self.predicate_classes is not None:
            fields_names = list(map(lambda f: f.name, self.parser.fields))
            for predicate_builder in self.predicate_classes:
                # Convert field name into indexes for the record predicate
                record_indexes = [
                    fields_names.index(requested_field)
                    for requested_field in predicate_builder.record_fields()
                ]
                try:
                    predicate = predicate_builder(
                        record_indexes,
                        # Extract args from the parameters
                        *[kwargs.pop(p) for p in predicate_builder.parameters()],
                    )
                    predicates.append(predicate)
                    logger.debug(
                        "Added predicate over parameters %s",
                        predicate_builder.parameters(),
                    )
                except KeyError:
                    logger.debug(
                        "Predicate build skipped, missing one of the following parameters %s",
                        predicate_builder.parameters(),
                    )

        df = self.discoverer.to_dataframe(
            predicates=predicates,
            stat_fields=stat_fields,
            enable_layouts=self.enable_layouts,
            **{k: kwargs[k] for k in kwargs if k in self.listing_parameters},
        )

        postprocesses = map(
            lambda item: item[1],
            filter(
                lambda item: item[0],
                [
                    (unmix and self.unmixer is not None, self.unmixer),
                    (deduplicate and self.deduplicator is not None, self.deduplicator),
                    (
                        sort and self.sort_keys is not None,
                        lambda df: df.sort_values(self.sort_keys, ignore_index=True),
                    ),
                ],
            ),
        )

        for postprocess in postprocesses:
            df = postprocess(df)

        return df

    def _query(self, **kwargs) -> xr_t.Dataset | None:
        """Query a dataset by reading selected files in file system.

        Raises
        ------
        LayoutMismatchError
            In case ``enable_layouts`` is True and a mismatch between the
            layouts and the actual files is detected
        ValueError
            In case if one unique and homogeneous subset could not be extracted
            from the files metadata table

        Returns
        -------
        A dataset containing the result of the query, or an None if there is
        nothing matching the query
        """
        # This docstring will be superseded by the metaclass
        bad_kwargs = [
            k
            for k in kwargs
            if k not in self.listing_parameters and k not in self.reading_parameters
        ]
        if bad_kwargs != []:
            raise ValueError(
                f"query() got unexpected keyword argument(s): {bad_kwargs}"
            )

        df = self._files(
            **{k: kwargs[k] for k in kwargs if k in self.listing_parameters},
            unmix=True,
            deduplicate=True,
            sort=True,
        )

        if len(df) == 0:
            return None

        # Reading parameters from the files metadata table
        reading_parameters = {
            k: df[k].values[0] for k in df if k in self.reading_parameters
        }
        # Reading parameters given by the user. These are given with a
        # lesser priority because user input may not be sanitized
        reading_parameters |= {
            k: kwargs[k] for k in kwargs if k in self.reading_parameters and k not in df
        }

        files = df["filename"].tolist()
        ds = self.reader.read(files=files, fs=self.fs, **reading_parameters)

        # Add another field in the dataset from the files metadata table
        if self.metadata_injection is not None:
            for variable, dimensions in self.metadata_injection.items():
                ds[variable] = (dimensions, df[variable])

        return ds

    def _variables_info(self, **kwargs) -> GroupMetadata | None:
        """Returns the variables metadata.

        Because the files collection may mix multiple subsets, we want to ensure
        that we return the variables of one subset only. The parameters of this
        method are the subset partitioning keys and can be given by the user to
        ensure a consistent set of variables. If the input parameters are not
        sufficient to unmix the subsets, the user will be notified with a
        ValueError

        Returns
        -------
            A GroupMetadata containing the variables, dimensions, attributes and
            subgroups. None is returned in case no files is found for the given
            subset

        Raises
        ------
        LayoutMismatchError
            In case ``enable_layouts`` is True and a mismatch between the
            layouts and the actual files is detected
        ValueError
            In case if one unique and homogeneous subset could not be extracted
            from the files metadata table
        """
        # This docstring will be superseded by the metaclass
        unknown = kwargs.keys() - (
            self.unmixer.partition_keys if self.unmixer is not None else set()
        )
        if len(unknown) > 0:
            msg = f"{inspect.stack()[0][3]} got unexpected keyword arguments {unknown}"
            raise TypeError(msg)

        df = self._files(**kwargs, unmix=True)
        if len(df.filename) == 0:
            warnings.warn('No files found with current filters "%s"' % kwargs)
            return
        file = df.filename[0]
        return self.reader.metadata(file, fs=self.fs)

    def _map(
        self, func: tp.Callable[[xr_t.Dataset, dict[str, tp.Any]], tp.Any], **kwargs
    ) -> dask.bag.core.Bag:
        """Map a function over dataset extracted from the files.

        Parameters
        ----------
        func
            Callable that works on a xarray dataset.

        Raises
        ------
        NotImplementedError
            In case dask is not available
        LayoutMismatchError
            In case ``enable_layouts`` is True and a mismatch between the
            layouts and the actual files is detected
        """
        try:
            import dask.bag.core
        except ImportError as exc:
            msg = "Install dask package to map a function over a files " "collection"
            raise NotImplementedError(msg) from exc

        df = self._files(
            **{k: kwargs[k] for k in kwargs if k in self.listing_parameters},
            unmix=True,
            deduplicate=True,
            sort=True,
        )

        # Reading parameters from the files metadata table and from the user.
        reading_parameters = {
            k: df[k].values[0] for k in df if k in self.reading_parameters
        } | {
            k: kwargs[k] for k in kwargs if k in self.reading_parameters and k not in df
        }

        def wrapped(record: dict[str, tp.Any]):
            ds = self.reader.read(
                files=[record.filename], fs=self.fs, **reading_parameters
            )
            return func(ds, record)

        bag = dask.bag.core.from_sequence(df.to_records())
        return bag.map(wrapped)


@dc.dataclass
class SubsetsUnmixer:
    partition_keys: tuple[str, ...] | dict[str, tp.Callable | None]
    auto_pick_last: tuple[str, ...] = dc.field(default_factory=tuple)

    def __call__(self, df: pda.DataFrame) -> pda.DataFrame:
        if len(df) == 0:
            return df

        try:
            subsets = df.groupby(
                [
                    (
                        df[partition_key].apply(transform)
                        if transform is not None
                        else df[partition_key]
                    )
                    for partition_key, transform in self.partition_keys.items()
                ]
            )
        except AttributeError:
            # We have a tuple
            grouping_keys = list(self.partition_keys)
            subsets = df.groupby(
                # In pandas 2, a list with one single element gave a scalar keys
                # for the groups. In pandas 3, a future warning is raised,
                # asking to give either a single key or a list with more than
                # one key
                grouping_keys if len(grouping_keys) > 1 else grouping_keys[0],
                # Pandas 3 tries to sort the groups, which can raise an error if
                # a column cannot be ordered. We don't need this sort so we
                # disable it
                sort=False,
            )

        # Pick one subset using panda duplicate handling
        subset_names = [
            (group,) if len(self.partition_keys) == 1 else group
            for group in subsets.groups.keys()
        ]
        df_subsets = pda.DataFrame.from_records(
            subset_names, columns=self.partition_keys
        )

        # Sort the dataframe containing the subset using the auto_pick_last keys
        # Unique records of manual_pick keys will be chosen relying on this sort
        if len(self.auto_pick_last) > 0:
            sort_keys = list(self.auto_pick_last)
            df_subsets.sort_values(sort_keys, inplace=True)

        # Because of the previous sort, if we ask pandas to drop the duplicates
        # keeping the last one, this will automatically choose the latest values
        # for the auto_pick_last keys
        manual_pick = list(set(self.partition_keys) - set(self.auto_pick_last))
        if len(manual_pick) > 0:
            df_subsets = df_subsets.drop_duplicates(manual_pick, keep="last")
            if len(df_subsets) > 1:
                ambiguity = {
                    key: df_subsets[key].unique().tolist()
                    for key in manual_pick
                    if len(df_subsets[key].unique()) > 1
                }
                msg = (
                    "Subsets could not be unmixed, the following keys are "
                    f"duplicated and should be fixed manually: {ambiguity}"
                )
                raise ValueError(msg)

        group_name = tuple(df_subsets.to_records(index=False)[-1])
        group_name = group_name if len(group_name) > 1 else group_name[0]
        logger.debug("Subset selected %s", group_name)
        return subsets.get_group(group_name)

    @property
    def keys(self) -> set[str]:
        return set(self.partition_keys) | set(self.auto_pick_last)


@dc.dataclass
class Deduplicator:
    unique: tuple[str, ...]
    auto_pick_last: tuple[str, ...] = dc.field(default_factory=tuple)

    def __call__(self, df: pda.DataFrame) -> pda.DataFrame:
        # Auto-deduplication using sort
        df.sort_values([*self.unique, *self.auto_pick_last], inplace=True)
        df = df.drop_duplicates(list(self.unique), keep="last")
        df.reset_index(inplace=True, drop=True)
        return df

    @property
    def keys(self) -> set[str]:
        return set(self.unique) | set(self.auto_pick_last)


class IPredicate(abc.ABC):
    """Interface for defining a complex predicate.

    This predicate will be used to filter records from file names listing and
    parsing.

    Attributes
    ----------
    indexes
        Attributes
    *args
        Any input that will be used to create the predicate
    """

    @abc.abstractmethod
    def __call__(self, record: tuple[tp.Any, ...]) -> bool:
        """Call the predicate.

        Parameters
        ----------
        record
            The record to filter

        Returns
        -------
        result
            True if the record complies with the criteria given by this
            predicate
        """

    @classmethod
    @abc.abstractmethod
    def record_fields(cls) -> tuple[str, ...]:
        """Record fields needed by the predicate."""

    @classmethod
    @abc.abstractmethod
    def parameters(cls) -> tuple[str, ...]:
        """Initialization parameters name for the class."""

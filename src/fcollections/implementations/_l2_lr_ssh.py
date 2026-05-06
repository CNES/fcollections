from __future__ import annotations

import dataclasses as dc
import functools
import re
from copy import copy
from enum import Enum, auto
from typing import TYPE_CHECKING

import fcollections.core as oct_io
from fcollections.core import (
    Deduplicator,
    FileNameConvention,
    FileNameFieldEnum,
    FileNameFieldInteger,
    FileNameFieldPeriod,
    FileNameFieldString,
    FilesDatabase,
    ICodec,
    Layout,
    PeriodMixin,
    SubsetsUnmixer,
)

from ._definitions._constants import DESCRIPTIONS
from ._definitions._swot import ProductSubset
from ._predicates import SwotPhasePredicate
from ._readers import SwotReaderL2LRSSH

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np_t


SWOT_L2_PATTERN = re.compile(
    r"SWOT_L2_LR_SSH_(?P<subset>.*)_(?P<cycle_number>\d{3})_(?P<pass_number>\d{3})_"
    r"(?P<time>\d{8}T\d{6}_\d{8}T\d{6})_(?P<version>P[I|G][A-Z]\d{1}_\d{2}).nc"
)


class Timeliness(Enum):
    """Timeliness of the SWOT L2_LR_SSH products."""

    I = auto()
    """Forward data."""
    G = auto()
    """Reprocessed data."""


@functools.total_ordering
@dc.dataclass
class L2Version:
    """Represents a L2 Version of half orbits and enables version comparison.

    A L2 Version is parsed from a string in format
    <CRID_version>_<product counter>.
    """

    temporality: Timeliness | None = None
    baseline: str | None = None
    minor_version: int | None = None
    product_counter: int | None = None
    ignore_product_counter_in_eq_check: bool = False

    def __eq__(self, other: L2Version):
        if not isinstance(other, L2Version):
            return False

        eq_res: bool = (
            self.baseline == other.baseline
            and self.temporality == other.temporality
            and self.minor_version == other.minor_version
        )

        if not self.ignore_product_counter_in_eq_check:
            # product counter check is intended (default behavior)
            eq_res &= self.product_counter == other.product_counter

        return eq_res

    @property
    def is_null(self):
        """True if all attrs but 'ignore_product_counter_in_eq_check' are
        None."""
        return all(
            [
                getattr(self, attr) is None
                for attr in [
                    "temporality",
                    "baseline",
                    "minor_version",
                    "product_counter",
                ]
            ]
        )

    @staticmethod
    def _are_orderable_members(left: L2Version, right: L2Version) -> bool:
        """Tell if inequality members can be ordered.

        Based on their 'baseline' and/or 'temporality' attributes values.

        Parameters
        ----------
        left
            leftmost member of the inequality
        right
            leftmost member of the inequality

        Return
        ------
        :
            False if:
            - right and/or left member has baseline and/or temporality attributes set to None
            - OR left member is not a L2Version
            True otherwise

        Note
        ----
        All ordering comparisions operators of this class handle non-orderable
        CRIDVersion the exact same way: return False as a comparison result.
        This is why we avoid code duplication with this method.

        This prevents errors like: `TypeError: '<' not supported between instances of 'NoneType' and 'str'`
        """
        if (
            not isinstance(right, L2Version)
            or left.baseline is None
            or left.temporality is None
            or right.baseline is None
            or right.temporality is None
        ):
            return False
        return True

    def _lt_operator_core(self, other: L2Version):
        """Core of the __lt__ operator.

        From this core function we can build other ordering comparison operators:
        - __lt__(a, b) -> _lt_operator_core(a, b)
        - __le__(a, b) -> not(_lt_operator_core(b, a))
        - __gt__(a, b) -> _lt_operator_core(b, a)
        - __ge__(a, b) -> not(_lt_operator_core(a, b))

        Note
        ----
        We need to implement each of these operators individualy (instead of letting python)
        default their logic from __lt__ operator, because we have to handle the case where
        self.baseline is None or self.temporality is None each time.
        """
        lt_res: bool = (
            self.baseline < other.baseline
            or (
                self.baseline == other.baseline
                and self.temporality.name > other.temporality.name
            )
            or (
                self.baseline == other.baseline
                and self.temporality == other.temporality
                and self.minor_version < other.minor_version
            )
            or (
                self.baseline == other.baseline
                and self.temporality == other.temporality
                and self.minor_version == other.minor_version
                and self.product_counter < other.product_counter
            )
        )

        return lt_res

    def __lt__(self, other: L2Version) -> bool:
        """Override '<' operator.

        Returns
        -------
        :
            True if members can be ordered and if self < other
            False otherwise

        Note
        ----
        if left or right member of the inequality has a 'baseline' and/or 'temporality'
        set to None, the inequality returns False.
        """
        if L2Version._are_orderable_members(self, other):
            return self._lt_operator_core(other)
        return False

    def __le__(self, other: L2Version) -> bool:
        """Override '<=' operator.

        Returns
        -------
        :
            True if members can be ordered and if self <= other
            False otherwise

        Note
        ----
        Like __lt__, if left or right member of the inequality has a 'baseline' and/or 'temporality'
        set to None, the inequality returns False. This is why we cannot simply let python default
        `__le__(a, b)` to `not(__lt__(b, a))`.
        """
        if L2Version._are_orderable_members(self, other):
            return not (other._lt_operator_core(self))
        return False

    def __gt__(self, other: L2Version) -> bool:
        """Override '>' operator.

        Returns
        -------
        :
            True if members can be ordered and if self > other
            False otherwise

        Note
        ----
        Like __lt__, if left or right member of the inequality has a 'baseline' and/or 'temporality'
        set to None, the inequality returns False. This is why we cannot simply let python default
        `__gt__(a, b)` to `__lt__(b, a)`.
        """
        if L2Version._are_orderable_members(self, other):
            return other._lt_operator_core(self)
        return False

    def __ge__(self, other: L2Version) -> bool:
        """Override '>=' operator.

        Returns
        -------
        :
            True if members can be ordered and if self >= other
            False otherwise

        Note
        ----
        Like __lt__, if left or right member of the inequality has a 'baseline' and/or 'temporality'
        set to None, the inequality returns False. This is why we cannot simply let python default
        `__ge__(a, b)` to `not(__lt__(a,b))`.
        """
        if L2Version._are_orderable_members(self, other):
            return not (self._lt_operator_core(other))
        return False

    def __repr__(self) -> str:
        temporality = self.temporality.name if self.temporality is not None else "?"
        baseline = self.baseline if self.baseline is not None else "?"
        minor_version = self.minor_version if self.minor_version is not None else "?"

        if self.product_counter is not None:
            return (
                f"P{temporality}{baseline}{minor_version}_{self.product_counter:0>2d}"
            )
        else:
            return f"P{temporality}{baseline}{minor_version}"

    def __hash__(self) -> int:
        # We need the hashing functionality to work nicely with pandas unique
        return hash(str(self))

    @staticmethod
    def from_bytes(
        version: bytes, ignore_product_counter_in_eq_check: bool = False
    ) -> L2Version | None:
        """Build a L2Version from bytes.

        Parameters
        ----------
        version:
            The CRID version from which we build the L2Version object.
        ignore_product_counter_in_eq_check:
            Set L2Version.product_counter to None, as we do not want to check it in
            the comparision operations.

        Returns
        -------
        :
            The L2Version object.

        Note
        ----
        Even when an `AttributeError` occures or input value is None, build a L2Version
        so that comparisons between np.array of L2Version do not fail with errors like:
        `TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'`.
        """
        parser = build_version_parser()
        if version in [None, "None"]:
            return L2Version(None, None, None, None)

        try:
            upstream_version = L2Version(
                *parser.parse(parser.match(bytes.decode(version)))
            )
            upstream_version.ignore_product_counter_in_eq_check = (
                ignore_product_counter_in_eq_check
            )
            return upstream_version
        except AttributeError:
            # version is invalid, still build a L2Version
            return L2Version(None, None, None, None)

    @staticmethod
    def from_string(
        version: str, ignore_product_counter_in_eq_check: bool = False
    ) -> L2Version | None:
        """Build a L2Version from str.

        Parameters
        ----------
        version:
            The CRID version from which we build the L2Version object.
        ignore_product_counter_in_eq_check:
            Set L2Version.product_counter to None, as we do not want to check it in
            the comparision operations.

        Returns
        -------
        :
            The L2Version object.

        Note
        ----
        Even when an `AttributeError` occures or input value is None, build a L2Version
        so that comparisons between np.array of L2Version do not fail with errors like:
        `TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'`.
        """
        parser = build_version_parser()
        if version in [None, "None"]:
            return L2Version(None, None, None, None)
        try:
            upstream_version = L2Version(*parser.parse(parser.match(version)))
            upstream_version.ignore_product_counter_in_eq_check = (
                ignore_product_counter_in_eq_check
            )
            return upstream_version
        except AttributeError:
            # version is invalid, still build a L2Version
            return L2Version(None, None, None, None)

    @staticmethod
    def from_bytes_array(
        versions: np_t.NDArray[bytes],
        ignore_product_counter_in_eq_check: bool = False,
    ) -> np_t.NDArray[object]:
        """Build a np.array of L2Version from an array of CRID versions as
        bytes.

        Parameters
        ----------
        versions:
            The array of CRID version from which we build the L2Version object.
        ignore_product_counter_in_eq_check:
            Set each L2Version.product_counter to None, as we do not want to check it in
            the comparision operations.

        Returns
        -------
        :
            The array of L2Version objects.
        """
        import numpy as np

        shape = versions.shape
        return np.array(
            [
                L2Version.from_bytes(v, ignore_product_counter_in_eq_check)
                for v in versions.ravel()
            ]
        ).reshape(shape)

    @staticmethod
    def from_string_array(
        versions: np_t.NDArray[str],
        ignore_product_counter_in_eq_check: bool = False,
    ) -> np_t.NDArray[object]:
        """Build a np.array of L2Version from an array of CRID versions as str.

        Parameters
        ----------
        versions:
            The array of CRID version from which we build the L2Version object.
        ignore_product_counter_in_eq_check:
            Set each L2Version.product_counter to None, as we do not want to check it in
            the comparision operations.

        Returns
        -------
        :
            The array of L2Version objects.
        """
        import numpy as np

        shape = versions.shape
        return np.array(
            [
                L2Version.from_string(v, ignore_product_counter_in_eq_check)
                for v in versions.ravel()
            ]
        ).reshape(shape)


def build_version_parser() -> FileNameConvention:
    """Build file name convention to parse CRID versions."""

    class _ExclamationMarkDecoder(ICodec):

        def decode(self, input_string):
            if input_string == "?":
                return None
            return super().decode(input_string)

    class _FileNameFieldEnum(_ExclamationMarkDecoder, FileNameFieldEnum):
        pass

    class _FileNameFieldInteger(_ExclamationMarkDecoder, FileNameFieldInteger):
        pass

    class _FileNameFieldString(_ExclamationMarkDecoder, FileNameFieldString):
        pass

    return FileNameConvention(
        regex=re.compile(
            r"P(?P<forward>I|G|\?)(?P<baseline>[A-Z]|\?)(?P<minor_version>[0-9]|\?)(_){0,1}(?P<product_counter>[0-9]{2}){0,1}$"
        ),
        fields=[
            _FileNameFieldEnum("forward", Timeliness),
            _FileNameFieldString("baseline"),
            _FileNameFieldInteger("minor_version"),
            FileNameFieldInteger("product_counter", default=None),
        ],
    )


class L2VersionField(oct_io.FileNameField):

    def __init__(self, name: str, ignore_product_counter: bool = False):
        super().__init__(
            name,
            L2Version(),
            description=(
                "Version of the L2_LR_SSH product, composed of a CRID and a "
                "product counter. The CRID can be further decomposed with the "
                "timeliness (I/G), the baseline (A/B/C...) and the minor "
                "version (a number) (ex. PIC0). The product counter is a number"
                "that increased when a half orbit has been regenerated for the "
                "same crid. This can happens if an anomaly is detected or if "
                "there is a change in the upstream data."
            ),
        )
        self.ignore_product_counter = ignore_product_counter

    def decode(self, input_string: str) -> L2Version:
        try:
            crid, product_counter = input_string.split("_")
            product_counter = int(product_counter)
        except ValueError:
            # Only the crid is present
            crid = input_string
            product_counter = None
        return L2Version(
            temporality=Timeliness[crid[1]],
            baseline=crid[2],
            minor_version=int(crid[3]),
            product_counter=product_counter,
        )

    def encode(self, data: L2Version) -> str:
        if self.ignore_product_counter:
            return str(L2Version(data.temporality, data.baseline, data.minor_version))
        return str(data)

    @property
    def test_description(self) -> str:
        # This is enforced in the equality of the L2Version object
        return (
            "As a L2Version field, this field can be tested by providing "
            "another L2Version instance. This instance can be partially set, with "
            "some missing attributes set to None. In this case, the check will be "
            "performed on these attributes only."
        )

    def test(self, reference: L2Version, tested: L2Version) -> bool:
        return all(
            [
                reference.temporality is None
                or reference.temporality == tested.temporality,
                reference.baseline is None or reference.baseline == tested.baseline,
                reference.minor_version is None
                or reference.minor_version == tested.minor_version,
                self.ignore_product_counter
                or reference.product_counter is None
                or reference.product_counter == tested.product_counter,
            ]
        )

    def sanitize(self, reference: str | L2Version) -> L2Version:
        if isinstance(reference, str):
            return L2Version.from_string(reference)
        return reference

    @property
    def type(self) -> type[L2Version]:
        return L2Version


class FileNameConventionSwotL2(FileNameConvention):
    """Swot LR L2 datafiles parser."""

    def __init__(self):
        super().__init__(
            regex=SWOT_L2_PATTERN,
            fields=[
                FileNameFieldInteger(
                    "cycle_number", description=DESCRIPTIONS["cycle_number"]
                ),
                FileNameFieldInteger(
                    "pass_number", description=DESCRIPTIONS["pass_number"]
                ),
                FileNameFieldPeriod(
                    "time", "%Y%m%dT%H%M%S", "_", description=DESCRIPTIONS["time"]
                ),
                FileNameFieldEnum(
                    "subset", ProductSubset, description=DESCRIPTIONS["subset"]
                ),
                L2VersionField("version"),
            ],
            generation_string="SWOT_L2_LR_SSH_{subset!f}_{cycle_number:>03d}_{pass_number:>03d}_{time!f}_{version!f}.nc",
        )


# In filenames, the version PID0_01 contains the crid and the product counter.
# In the layout, only the crid PID0 is present. When giving a reference, the
# user may give a product counter for filtering. This product counter should be
# ignored when testing occurs in the layout, else nothing will match this
# reference.
_ADAPTED_L2_FIELD: L2VersionField = copy(
    FileNameConventionSwotL2().get_field("version")
)
_ADAPTED_L2_FIELD.ignore_product_counter = True

AVISO_L2_LR_SSH_LAYOUT = Layout(
    [
        FileNameConvention(
            re.compile(r"^(?P<version>P[I|G][A-Z]\d{1})$"),
            [_ADAPTED_L2_FIELD],
            "{version!f}",
        ),
        FileNameConvention(
            re.compile(r"^(?P<subset>.*)$"),
            [FileNameConventionSwotL2().get_field("subset")],
            "{subset!f}",
        ),
        FileNameConvention(
            re.compile(r"^cycle_(?P<cycle_number>\d{3})$"),
            [FileNameConventionSwotL2().get_field("cycle_number")],
            "cycle_{cycle_number:0>3d}",
        ),
        FileNameConventionSwotL2(),
    ]
)


class BasicNetcdfFilesDatabaseSwotLRL2(FilesDatabase, PeriodMixin):
    """Database mapping to select and read Swot LR L2 Netcdf files in a local
    file system."""

    layouts = [Layout([FileNameConventionSwotL2()]), AVISO_L2_LR_SSH_LAYOUT]
    reader = SwotReaderL2LRSSH()
    sort_keys = "time"

    # These keys determines an homogeneous subset
    unmixer = SubsetsUnmixer(partition_keys=["subset"])
    # We expect multiple versions in an homogeneous subset. Only one half orbit
    # record is tolerated so we deduplicate the multiple version with an
    # autopick
    deduplicator = Deduplicator(
        unique=("cycle_number", "pass_number"), auto_pick_last=("version",)
    )

    # Convert phase filter to cycle_numbers filter
    predicate_classes = [SwotPhasePredicate]


try:
    from fcollections.implementations.optional import (
        GeoSwotReaderL2LRSSH,
        SwotGeometryPredicate,
    )

    class NetcdfFilesDatabaseSwotLRL2(BasicNetcdfFilesDatabaseSwotLRL2):
        reader = GeoSwotReaderL2LRSSH()
        predicate_classes = [SwotGeometryPredicate, SwotPhasePredicate]

except ImportError:
    import logging

    from ._definitions._constants import MISSING_OPTIONAL_DEPENDENCIES_MESSAGE

    logger = logging.getLogger(__name__)
    logger.info(MISSING_OPTIONAL_DEPENDENCIES_MESSAGE)

    NetcdfFilesDatabaseSwotLRL2 = BasicNetcdfFilesDatabaseSwotLRL2

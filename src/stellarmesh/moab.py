"""Stellarmesh MOAB/DAGMC models.

name: moab.py
author: Alex Koen, Paul Romano

desc: MOABModel class represents a MOAB model.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import warnings
from collections.abc import Iterable
from functools import cached_property
from typing import Final, Optional, Union

import numpy as np

from ._core import PathLike
from ._progress import log_progress, progress_heartbeat
from .mesh import Mesh

try:
    import gmsh
except ImportError as e:
    raise ImportError(
        "Gmsh not found. See Stellarmesh installation instructions."
    ) from e


try:
    import pymoab.core
    import pymoab.tag
    import pymoab.types
    from pymoab.rng import Range
except ImportError as e:
    raise ImportError(
        "PyMOAB not found. See Stellarmesh installation instructions."
    ) from e

logger = logging.getLogger(__name__)

#: DAGMC hard-codes the MOAB ``NAME`` tag to :data:`pymoab.types.NAME_TAG_SIZE`
#: bytes. Writing a wider ``NAME`` tag makes the file unreadable by DAGMC
#: ("Tag type in file does not match type in database for NAME"), so the
#: standard width is always used.
DAGMC_NAME_TAG_SIZE = pymoab.types.NAME_TAG_SIZE

#: Auxiliary tag holding the untruncated group name. DAGMC ignores it, while
#: stellarmesh uses it to preserve long CAD part and assembly names.
LONG_NAME_TAG_NAME = "STELLARMESH_NAME"

#: Byte width of :data:`LONG_NAME_TAG_NAME`.
LONG_NAME_TAG_SIZE = 512

#: Group name prefixes that must stay readable by DAGMC and therefore may not
#: be truncated.
UNTRUNCATABLE_PREFIXES = ("mat:", "boundary:")


class EntitySet:
    """A MOAB entity set."""

    _model: MOABModel
    handle: Final[np.uint64]

    @property
    def model(self) -> MOABModel:
        """Get owning MOABModel of this EntitySet."""
        return self._model

    def __init__(self, model: MOABModel, handle: np.uint64):
        """Initialize entity set.

        Args:
            model: MOAB model
            handle: Handle of entity set
        """
        self._model = model
        self.handle = handle

    def __eq__(self, other) -> bool:
        """Compare this entity with another."""
        return self.handle == other.handle

    def __hash__(self) -> int:
        """Return hash of entity set's handle."""
        return hash(self.handle)

    def __repr__(self) -> str:
        """String representation of entity set."""
        return f"<{type(self).__name__}(id={self.global_id})>"

    def _tag_get_data(self, tag: pymoab.tag.Tag):
        return self.model._core.tag_get_data(tag, self.handle, flat=True)[0]

    def _tag_set_data(self, tag: pymoab.tag.Tag, value):
        self.model._core.tag_set_data(tag, self.handle, value)

    @property
    def category(self) -> str:
        """Category for entity set."""
        return self._tag_get_data(self.model.category_tag)

    @category.setter
    def category(self, category: str):
        self._tag_set_data(self.model.category_tag, category)

    @property
    def global_id(self) -> int:
        """Global ID."""
        return self._tag_get_data(self.model.id_tag)

    @global_id.setter
    def global_id(self, value: int):
        self._tag_set_data(self.model.id_tag, value)

    @property
    def geom_dimension(self) -> int:
        """Geometry dimension."""
        return self._tag_get_data(self.model.geom_dimension_tag)

    @geom_dimension.setter
    def geom_dimension(self, dimension: int):
        self._tag_set_data(self.model.geom_dimension_tag, dimension)


class DAGMCGroup(EntitySet):
    """A DAGMC Group (used to assign material metadata)."""

    def __contains__(self, entity_set: EntitySet) -> bool:
        """Determine whether group contains a given entity set."""
        return any(vol.handle == entity_set.handle for vol in self.volumes)

    def __repr__(self) -> str:
        """String representation of group."""
        return f"<DAGMCGroup: {self.name})>"

    @property
    def name(self) -> str:
        """Name of the group.

        Returns the untruncated name when one was stored in the auxiliary
        :data:`LONG_NAME_TAG_NAME` tag, otherwise the MOAB ``NAME`` tag value.
        """
        model = self.model
        try:
            return str(
                model._core.tag_get_data(
                    model.long_name_tag, self.handle, flat=True
                )[0]
            )
        except RuntimeError:
            return model._core.tag_get_data(model.name_tag, self.handle, flat=True)[0]

    @name.setter
    def name(self, value: str):
        model = self.model
        name_tag = model.name_tag
        max_bytes = name_tag.get_length()
        encoded = value.encode()

        if len(encoded) >= max_bytes:
            if value.startswith(UNTRUNCATABLE_PREFIXES):
                raise ValueError(
                    f"DAGMC group name {value!r} must be shorter than the "
                    f"{max_bytes}-byte MOAB NAME tag. DAGMC reads this group "
                    "directly, so it cannot be truncated."
                )
            if len(encoded) >= LONG_NAME_TAG_SIZE:
                raise ValueError(
                    f"Group name {value!r} exceeds the "
                    f"{LONG_NAME_TAG_SIZE}-byte {LONG_NAME_TAG_NAME} tag."
                )
            short_value = model._unique_short_name(value, max_bytes - 1)
            logger.warning(
                f"Group name {value!r} exceeds the {max_bytes}-byte MOAB NAME "
                f"tag required by DAGMC. Storing {short_value!r} in NAME and "
                f"the full name in the {LONG_NAME_TAG_NAME} tag."
            )
            model._core.tag_set_data(name_tag, self.handle, short_value)
            model._core.tag_set_data(model.long_name_tag, self.handle, value)
            return

        model._core.tag_set_data(name_tag, self.handle, value)
        try:
            model._core.tag_delete_data(model.long_name_tag, self.handle)
        except RuntimeError:
            pass

    @property
    def volumes(self) -> list[DAGMCVolume]:
        """Get list of volumes contained in this group."""
        handles: Range = self.model._core.get_entities_by_type_and_tag(
            self.handle, pymoab.types.MBENTITYSET, [self.model.category_tag], ["Volume"]
        )
        return [DAGMCVolume(self.model, handle) for handle in handles]

    @property
    def surfaces(self) -> list[DAGMCSurface]:
        """Get list of surfaces contained in this group."""
        handles: Range = self.model._core.get_entities_by_type_and_tag(
            self.handle,
            pymoab.types.MBENTITYSET,
            [self.model.category_tag],
            ["Surface"],
        )
        return [DAGMCSurface(self.model, handle) for handle in handles]

    def add(self, entity_set: EntitySet):
        """Add entity set to the group.

        Args:
            entity_set: Entity set to add
        """
        self.model._core.add_entity(self.handle, entity_set.handle)

    def remove(self, entity_set: EntitySet):
        """Remove entity set from the group.

        Args:
            entity_set: Entity set to remove
        """
        self.model._core.remove_entity(self.handle, entity_set.handle)


class DAGMCEntitySet(EntitySet):
    """An entity set for a DAGMC topological surface or volume."""

    @property
    def model(self) -> DAGMCModel:
        """Get owning DAGMCModel of this DAGMCEntitySet."""
        return self._model  # type: ignore

    @property
    def groups(self) -> list[DAGMCGroup]:
        """Get list of groups containing this volume."""
        return [group for group in self.model.groups if self in group]


class DAGMCCurve(DAGMCEntitySet):
    """DAGMC curve entity."""

    @property
    def adjacent_surfaces(self) -> list[DAGMCSurface]:
        """Get adjacent surfaces.

        Returns:
            Adjacent surfaces.
        """
        parent_entities = self.model._core.get_parent_meshsets(self.handle)
        return [DAGMCSurface(self.model, e) for e in parent_entities]


class DAGMCSurface(DAGMCEntitySet):
    """DAGMC surface entity."""

    @property
    def forward_volume(self) -> Optional[DAGMCVolume]:
        """Volume with forward sense with respect to the surface."""
        return self.surf_sense[0]

    @forward_volume.setter
    def forward_volume(self, volume: DAGMCVolume):
        self.surf_sense = (volume, self.reverse_volume)

    @property
    def reverse_volume(self) -> Optional[DAGMCVolume]:
        """Volume with reverse sense with respect to the surface."""
        return self.surf_sense[1]

    @reverse_volume.setter
    def reverse_volume(self, volume: DAGMCVolume):
        self.surf_sense = (self.forward_volume, volume)

    @property
    def surf_sense(self) -> list[Optional[DAGMCVolume]]:
        """Surface sense data."""
        try:
            handles = self.model._core.tag_get_data(
                self.model.surf_sense_tag, self.handle, flat=True
            )
        except RuntimeError:
            return [None, None]

        return [
            DAGMCVolume(self.model, handle) if handle != 0 else None
            for handle in handles
        ]

    @surf_sense.setter
    def surf_sense(self, volumes: tuple[Optional[DAGMCVolume], Optional[DAGMCVolume]]):
        sense_data = [
            vol.handle if vol is not None else np.uint64(0) for vol in volumes
        ]
        self._tag_set_data(self.model.surf_sense_tag, sense_data)

        parents = self.model._core.get_parent_meshsets(self.handle)
        for parent in parents:
            if parent not in sense_data:
                # REVIEW (akoen): remove_parent_child only implemented in MOAB 5.6.0
                if hasattr(self.model._core, "remove_parent_child"):
                    self.model._core.remove_parent_child(parent.handle, self.handle)  # pyright: ignore[reportAttributeAccessIssue]
                logger.warning(
                    f"Surface has existing parent {parent} that cannot be removed in "
                    + "this version of MOAB."
                )
        # Establish parent-child relationships
        for vol in volumes:
            if vol is not None:
                self.model._core.add_parent_child(vol.handle, self.handle)

    @property
    def boundary(self) -> Optional[str]:
        """Name of the boundary condition assigned to this surface."""
        for group in self.groups:
            if group.name.startswith("boundary:"):
                return group.name[9:]
        return None

    @boundary.setter
    def boundary(self, name: str):
        existing_group = False
        for group in self.model.groups:
            if f"boundary:{name}" == group.name:
                if self in group:
                    return
                group.add(self)
                existing_group = True

            elif self in group and group.name.startswith("boundary:"):
                # Remove volume from existing group
                group.remove(self)

        if not existing_group:
            new_group = self.model.create_group(f"boundary:{name}")
            new_group.global_id = (
                max((g.global_id for g in self.model.groups), default=0) + 1
            )
            new_group.add(self)

    @property
    def adjacent_volumes(self) -> list[DAGMCVolume]:
        """Get adjacent volumes.

        Returns:
            Adjacent volumes.
        """
        parent_entities = self.model._core.get_parent_meshsets(self.handle)
        return [DAGMCVolume(self.model, e) for e in parent_entities]

    @property
    def triangles(self) -> Range:
        """Get range of triangle elements."""
        return self.model._core.get_entities_by_type(self.handle, pymoab.types.MBTRI)


class DAGMCVolume(DAGMCEntitySet):
    """DAGMC volume entity."""

    @property
    def adjacent_surfaces(self) -> list[DAGMCSurface]:
        """Get adjacent surfaces.

        Returns:
            Adjacent surfaces.
        """
        child_entities = self.model._core.get_child_meshsets(self.handle)
        return [DAGMCSurface(self.model, e) for e in child_entities]

    @property
    def material(self) -> Optional[str]:
        """Name of the material assigned to this volume."""
        for group in self.groups:
            if self in group and group.name.startswith("mat:"):
                return group.name[4:]
        return None

    @material.setter
    def material(self, name: str):
        existing_group = False
        for group in self.model.groups:
            if f"mat:{name}" == group.name:
                if self in group:
                    return
                group.add(self)
                existing_group = True

            elif self in group and group.name.startswith("mat:"):
                group.remove(self)

        if not existing_group:
            new_group = self.model.create_group(f"mat:{name}")
            new_group.global_id = (
                max((g.global_id for g in self.model.groups), default=0) + 1
            )
            new_group.add(self)

    @property
    def part(self) -> Optional[str]:
        """CAD part name assigned to this volume."""
        for group in self.groups:
            if self in group and group.name.startswith("part:"):
                return group.name[5:]
        return None

    @part.setter
    def part(self, name: str):
        existing_group = False
        for group in self.model.groups:
            if f"part:{name}" == group.name:
                if self in group:
                    return
                group.add(self)
                existing_group = True
            elif self in group and group.name.startswith("part:"):
                group.remove(self)

        if not existing_group:
            new_group = self.model.create_group(f"part:{name}")
            new_group.global_id = (
                max((g.global_id for g in self.model.groups), default=0) + 1
            )
            new_group.add(self)

    @property
    def bounding_box(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Axis-aligned bounding box for this volume.

        Returns:
            Two 3-tuples giving the minimum and maximum XYZ coordinates:
            ``((xmin, ymin, zmin), (xmax, ymax, zmax))``.

        Raises:
            ValueError: If no triangle vertices are found for this volume.
        """
        mins = []
        maxs = []

        for surface in self.adjacent_surfaces:
            triangles = surface.triangles
            if triangles.empty():
                continue

            vertices = np.unique(self.model._core.get_connectivity(triangles))
            if vertices.size == 0:
                continue

            coords = self.model._core.get_coords(vertices).reshape(-1, 3)
            mins.append(coords.min(axis=0))
            maxs.append(coords.max(axis=0))

        if not mins:
            raise ValueError(f"Volume {self.global_id} has no triangle vertices.")

        min_xyz = np.min(np.stack(mins), axis=0)
        max_xyz = np.max(np.stack(maxs), axis=0)
        return (tuple(min_xyz.tolist()), tuple(max_xyz.tolist()))


class MOABModel:
    """MOAB Model.

    This class holds a generic MOAB mesh, which could be a 2D surface mesh used
    in DAGMC, a 3D tetrahedral mesh, etc.
    """

    _core: pymoab.core.Core

    def __init__(self, core_or_file: Union[PathLike, pymoab.core.Core]):
        """Initialize model from a file or existing pymoab Core object.

        Args:
            core_or_file: path-like or Core object.
        """
        if isinstance(core_or_file, (str, os.PathLike)):
            core = pymoab.core.Core()
            core.load_file(str(core_or_file))
        elif isinstance(core_or_file, (pymoab.core.Core)):
            core = core_or_file
        else:
            raise TypeError("core_or_file is of invalid type.")
        self._core = core

    @cached_property
    def category_tag(self) -> pymoab.tag.Tag:
        """Category tag."""
        return self._core.tag_get_handle(
            pymoab.types.CATEGORY_TAG_NAME,
            pymoab.types.CATEGORY_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def name_tag(self) -> pymoab.tag.Tag:
        """Name tag.

        New models always use :data:`DAGMC_NAME_TAG_SIZE`, the width DAGMC
        expects. An existing (possibly legacy, wider) tag is reused so that
        files written by older versions remain readable.
        """
        try:
            return self._core.tag_get_handle(pymoab.types.NAME_TAG_NAME)
        except RuntimeError:
            return self._core.tag_get_handle(
                pymoab.types.NAME_TAG_NAME,
                DAGMC_NAME_TAG_SIZE,
                pymoab.types.MB_TYPE_OPAQUE,
                pymoab.types.MB_TAG_SPARSE,
                create_if_missing=True,
            )

    @cached_property
    def long_name_tag(self) -> pymoab.tag.Tag:
        """Auxiliary tag holding untruncated group names."""
        return self._core.tag_get_handle(
            LONG_NAME_TAG_NAME,
            LONG_NAME_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    def _unique_short_name(self, value: str, max_bytes: int) -> str:
        """Truncate a group name to max_bytes, avoiding NAME tag collisions."""
        existing = set()
        for group in getattr(self, "groups", []):
            try:
                existing.add(
                    self._core.tag_get_data(self.name_tag, group.handle, flat=True)[0]
                )
            except RuntimeError:
                continue

        def _truncate(text: str) -> str:
            encoded = text.encode()[:max_bytes]
            return encoded.decode(errors="ignore")

        candidate = _truncate(value)
        if candidate not in existing:
            return candidate
        for index in range(1, 1000):
            suffix = f"~{index}"
            candidate = _truncate(value)[: max_bytes - len(suffix)] + suffix
            if candidate not in existing:
                return candidate
        raise ValueError(f"Could not derive a unique short name for {value!r}.")

    @cached_property
    def id_tag(self) -> pymoab.tag.Tag:
        """Global ID tag."""
        # Default tag, does not need to be created
        return self._core.tag_get_handle(pymoab.types.GLOBAL_ID_TAG_NAME)

    @cached_property
    def geom_dimension_tag(self) -> pymoab.tag.Tag:
        """Geometry dimension tag."""
        return self._core.tag_get_handle(
            pymoab.types.GEOM_DIMENSION_TAG_NAME,
            1,
            pymoab.types.MB_TYPE_INTEGER,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def surf_sense_tag(self) -> pymoab.tag.Tag:
        """Surface sense tag."""
        return self._core.tag_get_handle(
            "GEOM_SENSE_2",
            2,
            pymoab.types.MB_TYPE_HANDLE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def faceting_tol_tag(self) -> pymoab.tag.Tag:
        """Faceting tolerance tag."""
        return self._core.tag_get_handle(
            "FACETING_TOL",
            1,
            pymoab.types.MB_TYPE_DOUBLE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @property
    def root_set(self) -> np.uint64:
        """Get handle of MOAB root entity set."""
        return self._core.get_root_set()

    @property
    def tets(self) -> Range:
        """Get range of tetrahedral elements."""
        return self._core.get_entities_by_type(
            self.root_set, pymoab.types.MBTET, recur=True
        )

    @property
    def triangles(self) -> Range:
        """Get range of triangle elements."""
        return self._core.get_entities_by_type(
            self.root_set, pymoab.types.MBTRI, recur=True
        )

    def _add_nodes(self) -> dict[int, int]:
        """Generic node addition logic shared by all MOAB-based models."""
        node_tags, coords, _ = gmsh.model.mesh.get_nodes()
        if np.isnan(coords).any():
            raise ValueError("Mesh coordinates contain NaNs.")
        if np.isinf(coords).any():
            raise ValueError("Mesh coordinates contain infinite values.")

        moab_vertices = self._core.create_vertices(coords)
        self._core.tag_set_data(self.id_tag, moab_vertices, node_tags.astype(np.int32))  # pyright: ignore[reportAttributeAccessIssue]

        node_tag_map = dict(zip(node_tags, moab_vertices, strict=True))
        if len(node_tag_map) != len(node_tags):
            raise ValueError("Duplicate node tags found.")
        return node_tag_map

    def _create_elements(
        self, dim: int, tag: int, node_tag_map: dict[int, int]
    ) -> Range:
        """Generic element creation for any dimension (2D or 3D)."""
        element_types, _, node_tags_list = gmsh.model.mesh.get_elements(dim, tag)
        all_new_handles = []

        for elem_type, node_tags in zip(element_types, node_tags_list, strict=True):
            # Map Gmsh types to MOAB types
            if elem_type == 2:  # Triangle
                moab_type, nodes_per_elem = pymoab.types.MBTRI, 3
            elif elem_type == 4:  # Tet
                moab_type, nodes_per_elem = pymoab.types.MBTET, 4
            elif elem_type == 5:  # Hex
                moab_type, nodes_per_elem = pymoab.types.MBHEX, 8
            else:
                continue

            conn = np.array(
                [node_tag_map[t] for t in node_tags], dtype=np.uint64
            ).reshape(-1, nodes_per_elem)

            all_new_handles = [self._core.create_element(moab_type, c) for c in conn]

        return Range(all_new_handles)

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> MOABModel:
        """Create MOAB model from mesh.

        Args:
            mesh: Mesh from which to build MOAB mesh.

        Returns:
            Initialized model.
        """
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=True) as mesh_file:
            with progress_heartbeat(logger, "Converting mesh to MOAB"), mesh:
                gmsh.write(mesh_file.name)
            return cls(mesh_file.name)

    @classmethod
    def read_file(cls, h5m_file: PathLike) -> MOABModel:
        """Initialize model from .h5m file.

        Args:
            h5m_file: File to load.

        Returns:
            Initialized model.
        """
        warnings.warn(
            f"The read_file method is deprecated. Use {cls.__name__}(...) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls(h5m_file)

    def write(self, filename: PathLike):
        """Write MOAB model to .h5m, .vtk, or other file.

        Args:
            filename: Filename with format-appropriate extension.
        """
        logger.info(f"Writing MOAB mesh to {filename!s}")
        self._core.write_file(str(filename))


class DAGMCModel(MOABModel):
    """DAGMC Model."""

    @property
    def part_to_volume_ids(self) -> dict[str, list[int]]:
        """Map CAD part names to DAGMC volume global IDs."""
        mapping: dict[str, list[int]] = {}
        for group in self.groups:
            if not group.name.startswith("part:"):
                continue
            mapping[group.name[5:]] = sorted(
                volume.global_id for volume in group.volumes
            )
        return mapping

    @property
    def material_to_volume_ids(self) -> dict[str, list[int]]:
        """Map material names to DAGMC volume global IDs.

        This mapping is derived from DAGMC ``mat:<name>`` groups in the model.
        The returned IDs correspond to OpenMC DAGMC cell IDs.

        Returns:
            Dictionary mapping material name (without ``mat:`` prefix) to
            sorted list of volume ``GLOBAL_ID`` values.
        """
        mapping: dict[str, list[int]] = {}
        for group in self.groups:
            if not group.name.startswith("mat:"):
                continue
            mapping[group.name[4:]] = sorted(
                volume.global_id for volume in group.volumes
            )
        return mapping

    @property
    def assembly_to_volume_ids(self) -> dict[str, list[int]]:
        """Map assembly name to contained DAGMC volume global IDs."""
        mapping: dict[str, list[int]] = {}
        for group in self.groups:
            if not group.name.startswith("assembly:"):
                continue
            mapping[group.name[9:]] = sorted(
                volume.global_id for volume in group.volumes
            )
        return mapping

    @property
    def name_to_volume_ids(self) -> dict[str, list[int]]:
        """Map any part, assembly, or material name to DAGMC volume IDs."""
        mapping: dict[str, list[int]] = {}
        sources = (
            ("part", self.part_to_volume_ids),
            ("assembly", self.assembly_to_volume_ids),
            ("material", self.material_to_volume_ids),
        )
        origins: dict[str, str] = {}
        for origin, source in sources:
            for name, volume_ids in source.items():
                if name in mapping and mapping[name] != volume_ids:
                    raise ValueError(
                        f"Name {name!r} is used by both {origins[name]} and "
                        f"{origin} metadata with different volumes. Use the "
                        "explicit mapping properties to disambiguate it."
                    )
                mapping[name] = list(volume_ids)
                origins.setdefault(name, origin)
        return mapping

    def volume_ids(self, name: str) -> list[int]:
        """Return volume IDs for a part, assembly, or material name."""
        part_ids = self.part_to_volume_ids.get(name)
        material_ids = self.material_to_volume_ids.get(name)
        assembly_ids = self.assembly_to_volume_ids.get(name)
        matches = [
            (kind, ids)
            for kind, ids in (
                ("part", part_ids),
                ("assembly", assembly_ids),
                ("material", material_ids),
            )
            if ids is not None
        ]
        if not matches:
            raise KeyError(f"Unknown material, part, or assembly name: {name}")
        if any(ids != matches[0][1] for _, ids in matches[1:]):
            kinds = ", ".join(kind for kind, _ in matches)
            raise ValueError(
                f"Name {name!r} has different volume mappings as {kinds}. "
                "Use the explicit mapping properties to disambiguate it."
            )
        return list(matches[0][1])

    def bounding_box(
        self, volume_ids_or_name: Union[str, Iterable[int]]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get combined bounding box for selected DAGMC volumes.

        Args:
            volume_ids_or_name: A part, assembly, or material name, or an iterable
                of volume ``GLOBAL_ID`` values.

        Returns:
            Two 3-tuples giving the minimum and maximum XYZ coordinates:
            ``((xmin, ymin, zmin), (xmax, ymax, zmax))``.

        Raises:
            KeyError: If ``volume_ids_or_name`` is a name not found.
            ValueError: If no matching volumes are found for provided IDs.
        """
        if isinstance(volume_ids_or_name, str):
            volume_ids = set(self.volume_ids(volume_ids_or_name))
        else:
            volume_ids = set(volume_ids_or_name)
            if not volume_ids:
                raise ValueError("No volume IDs were provided.")
        if isinstance(volume_ids_or_name, str) and not volume_ids:
            raise ValueError(f"Name '{volume_ids_or_name}' has no associated volumes.")

        selected_volumes = [v for v in self.volumes if v.global_id in volume_ids]
        found_volume_ids = {v.global_id for v in selected_volumes}
        missing_volume_ids = volume_ids - found_volume_ids
        if missing_volume_ids:
            missing_str = ", ".join(str(v) for v in sorted(missing_volume_ids))
            raise ValueError(f"Unknown volume IDs: {missing_str}")

        mins = []
        maxs = []
        for volume in selected_volumes:
            min_xyz, max_xyz = volume.bounding_box
            mins.append(np.array(min_xyz))
            maxs.append(np.array(max_xyz))

        combined_min = np.min(np.stack(mins), axis=0)
        combined_max = np.max(np.stack(maxs), axis=0)
        return (tuple(combined_min.tolist()), tuple(combined_max.tolist()))

    def create_group(self, group_name: str) -> DAGMCGroup:
        """Create new group.

        Args:
            group_name: Name assigned to the new group

        Returns:
            Group object.
        """
        group = DAGMCGroup(self, self._core.create_meshset())
        group.category = "Group"
        group.name = group_name
        return group

    def create_volume(self, global_id: Optional[int] = None) -> DAGMCVolume:
        """Create new volume.

        Args:
            global_id: Global ID.

        Returns:
            Volume object.
        """
        volume = DAGMCVolume(self, self._core.create_meshset())
        volume.geom_dimension = 3
        volume.category = "Volume"
        if global_id is not None:
            volume.global_id = global_id
        return volume

    def create_surface(self, global_id: Optional[int] = None) -> DAGMCSurface:
        """Create new surface.

        Args:
            global_id: Global ID.

        Returns:
            Surface object.
        """
        surface = DAGMCSurface(self, self._core.create_meshset())
        surface.geom_dimension = 2
        surface.category = "Surface"
        if global_id is not None:
            surface.global_id = global_id
        return surface

    def create_curve(self, global_id: Optional[int] = None) -> DAGMCCurve:
        """Create new curve.

        Args:
            global_id: Global ID.

        Returns:
            curve object.
        """
        curve = DAGMCCurve(self, self._core.create_meshset())
        curve.geom_dimension = 1
        curve.category = "Curve"
        if global_id is not None:
            curve.global_id = global_id
        return curve

    @property
    def groups(self) -> list[DAGMCGroup]:
        """Get list of groups."""
        group_handles: Range = self._core.get_entities_by_type_and_tag(
            self.root_set,
            pymoab.types.MBENTITYSET,
            [self.category_tag],
            ["Group"],
        )
        return [DAGMCGroup(self, handle) for handle in group_handles]

    @staticmethod
    def make_watertight(
        input_filename: PathLike,
        output_filename: PathLike,
        binary_path: str = "make_watertight",
    ) -> bool:
        """Make mesh watertight.

        Args:
            input_filename: Input .h5m filename.
            output_filename: Output watertight .h5m filename.
            binary_path: Path to make_watertight or default to find in path. Defaults to
            "make_watertight".
        """
        subprocess.run(
            [binary_path, str(input_filename), "-o", str(output_filename)],
            check=True,
        )

    @staticmethod
    def check_overlap(
        input_filename: PathLike,
        binary_path: str = "overlap_check",
        points_per_edge: int = 0,
        num_threads: int = 1,
    ) -> bool:
        """Check mesh for overlaps.

        Args:
            input_filename: Input .h5m filename.
            binary_path: Path to overlap_check or default to find in path. Defaults to
            "overlap_check".
            points_per_edge: Number of evenly-spaced points to test on each triangle
                edge. If points_per_edge=0, only triangle vertex locations are checked.
                Defaults to 0.
            num_threads: Number of threads.

        Returns:
            True if no overlaps are found, else False.
        """
        out = subprocess.run(
            [
                binary_path,
                str(input_filename),
                "-p",
                str(points_per_edge),
                "-t",
                str(num_threads),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return "No overlaps were found." in out.stdout.splitlines()

    @staticmethod
    def check_watertight(
        input_filename: PathLike,
        binary_path: str = "check_watertight",
    ):
        """Check mesh for watertightness.

        Args:
            input_filename: Input .h5m filename.
            binary_path: Path to overlap_check or default to find in path. Defaults to
            "check_watertight".

        Returns:
            True if mesh is watertight, else False.
        """
        out = subprocess.run(
            [binary_path, str(input_filename)],
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = out.stdout.splitlines()
        for line in output_lines:
            if ("leaky surface ids=" in line) or ("leaky volume ids=" in line):
                return line.strip().endswith("=")
        return False

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
    ) -> DAGMCModel:
        """Compose DAGMC MOAB .h5m file from mesh.

        Args:
            mesh: Mesh from which to build DAGMC geometry.
        """
        core = pymoab.core.Core()
        model = cls(core)

        with progress_heartbeat(logger, "Converting mesh to DAGMC"), mesh:
            if gmsh.model.mesh.get_elements(3)[1]:
                logger.warning("Discarding volume elements from mesh.")

            gmsh.model.mesh.removeDuplicateNodes()

            node_tag_map = model._add_nodes()
            surface_map = model._add_surfaces(mesh, node_tag_map)
            model._add_volumes(mesh, surface_map)
            model._finalize_file_set()

            return model

    def _add_surfaces(
        self, mesh: Mesh, node_tag_map: dict[int, int]
    ) -> dict[int, DAGMCSurface]:
        """Add surfaces to MOAB model.

        Return map from Gmsh tag to MOAB handle.
        """
        surface_map: dict[int, DAGMCSurface] = {}
        surface_dimtags = gmsh.model.get_entities(2)
        surface_tags = [s[1] for s in surface_dimtags]
        logger.debug(f"Mesh has {len(surface_tags)} surfaces")

        for i, surface_tag in enumerate(surface_tags, start=1):
            surface_set = self.create_surface(surface_tag)
            surface_map[surface_tag] = surface_set

            if (
                bc := mesh.entity_metadata(2, surface_tag).boundary_condition
            ) is not None:
                surface_set.boundary = bc

            self._create_surface_elements(surface_tag, surface_set, node_tag_map)
            self._create_volume_friend_for_lonely_surfaces(surface_tag, surface_set)
            log_progress(logger, "Creating DAGMC surfaces", i, len(surface_tags))

        return surface_map

    def _create_surface_elements(
        self, surface_tag: int, surface_set: DAGMCSurface, node_tag_map: dict[int, int]
    ):
        """Process elements for a single surface."""
        triangles = self._create_elements(2, surface_tag, node_tag_map)
        if not triangles:
            raise RuntimeError(f"Surface {surface_tag} has no elements")

        self._core.add_entities(surface_set.handle, triangles)
        # Add vertices to the set as well (topologically required by some tools?)
        adj_verts = self._core.get_adjacencies(triangles, 0, create_if_missing=False)
        self._core.add_entities(surface_set.handle, adj_verts)

    def _create_volume_friend_for_lonely_surfaces(
        self, surface_tag: int, surface_set: DAGMCSurface
    ):
        """Handle surfaces without attached volumes by creating void volumes."""
        volume_adjacencies_dimtags = gmsh.model.get_adjacencies(2, surface_tag)[0]
        if len(volume_adjacencies_dimtags) == 0:
            logger.warning(
                "DAGMC does not support surfaces without attached "
                "volumes. Creating a void volume for surface "
                f"{surface_tag}."
            )
            existing_vol_tags = [v[1] for v in gmsh.model.get_entities(3)]
            new_id = max(existing_vol_tags, default=0) + 1
            volume_set = self.create_volume(new_id)
            volume_set.material = "void"
            surface_set.forward_volume = volume_set

    def _add_volumes(self, mesh: Mesh, surface_map: dict[int, DAGMCSurface]):
        """Add volumes and set up surface sense metadata."""
        volume_dimtags = gmsh.model.get_entities(3)
        volume_tags = [v[1] for v in volume_dimtags]
        volume_map: dict[int, DAGMCVolume] = {}
        logger.debug(f"Mesh has {len(volume_tags)} volumes")

        for i, volume_tag in enumerate(volume_tags, start=1):
            volume_set = self.create_volume(volume_tag)
            volume_map[volume_tag] = volume_set
            metadata = mesh.entity_metadata(3, volume_tag)
            mat_name = metadata.material
            volume_set.material = mat_name
            if (part_name := metadata.part) is not None:
                volume_set.part = part_name
            log_progress(logger, "Creating DAGMC volumes", i, len(volume_tags))

        next_group_id = max((g.global_id for g in self.groups), default=0) + 1
        for dim, physical_tag in gmsh.model.get_physical_groups(3):
            group_name = gmsh.model.get_physical_name(dim, physical_tag)
            if not group_name.startswith("assembly:"):
                continue
            group = self.create_group(group_name)
            group.global_id = next_group_id
            next_group_id += 1
            for volume_tag in gmsh.model.get_entities_for_physical_group(
                dim, physical_tag
            ):
                group.add(volume_map[volume_tag])

        for surface_tag, surface in surface_map.items():
            metadata = mesh.entity_metadata(2, surface_tag)
            if (forward_vol_tag := metadata.forward_volume) is not None:
                forward_vol = volume_map.get(forward_vol_tag)
                assert forward_vol is not None
                surface.forward_volume = forward_vol
            if (reverse_vol_tag := metadata.reverse_volume) is not None:
                reverse_vol = volume_map.get(reverse_vol_tag)
                assert reverse_vol is not None
                surface.reverse_volume = reverse_vol

        # Warn on empty volumes
        for volume in volume_map.values():
            if not self._core.get_child_meshsets(volume.handle):
                logger.error(f"Volume {volume.global_id} has no assigned surfaces.")

    def _finalize_file_set(self):
        """Create file set and set global tags."""
        all_entities = self._core.get_entities_by_handle(0)
        file_set = self._core.create_meshset()
        # TODO(akoen): faceting tol set to a random value
        # https://github.com/Thea-Energy/neutronics-cad/issues/5
        self._core.tag_set_data(self.faceting_tol_tag, file_set, 0.1)
        self._core.add_entities(file_set, all_entities)

    @classmethod
    def make_from_mesh(cls, mesh: Mesh) -> DAGMCModel:
        """Compose DAGMC MOAB .h5m file from mesh.

        Args:
            mesh: Mesh from which to build DAGMC geometry.
        """
        warnings.warn(
            "The make_from_mesh method is deprecated. Use from_mesh instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_mesh(mesh)

    def _get_entities_of_geom_dimension(self, dim: int) -> list[np.uint64]:
        return self._core.get_entities_by_type_and_tag(
            0, pymoab.types.MBENTITYSET, [self.geom_dimension_tag], [dim]
        )

    @property
    def curves(self) -> list[DAGMCCurve]:
        """Get curves in this model.

        Returns:
            Curve.
        """
        curve_handles = self._get_entities_of_geom_dimension(1)
        return [DAGMCCurve(self, h) for h in curve_handles]

    @property
    def surfaces(self) -> list[DAGMCSurface]:
        """Get surfaces in this model.

        Returns:
            Surfaces.
        """
        surface_handles = self._get_entities_of_geom_dimension(2)
        return [DAGMCSurface(self, h) for h in surface_handles]

    @property
    def volumes(self) -> list[DAGMCVolume]:
        """Get volumes in this model.

        Returns:
            Volumes.
        """
        volume_handles = self._get_entities_of_geom_dimension(3)
        return [DAGMCVolume(self, h) for h in volume_handles]


class MOABVolumeModel(MOABModel):
    """A MOAB model consisting of 3D volume elements, typically for tallies."""

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> MOABVolumeModel:
        """Create Volume Mesh MOAB file from mesh."""
        core = pymoab.core.Core()
        model = cls(core)
        with mesh:
            gmsh.model.mesh.removeDuplicateNodes()
            node_map = model._add_nodes()

            # Simply add all 3D elements to the root set
            for _, tag in gmsh.model.get_entities(3):
                model._create_elements(3, tag, node_map)

        return model

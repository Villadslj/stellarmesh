"""Stellarmesh geometry.

name: geometry.py
author: Alex Koen

desc: Geometry class represents a CAD geometry to be meshed.
"""

from __future__ import annotations

import logging
import warnings
from typing import (
    Optional,
    Protocol,
    Sequence,
    Type,
    overload,
)

try:
    from OCP.Bnd import Bnd_Box  # pyright: ignore[reportMissingModuleSource]
    from OCP.BOPAlgo import (  # pyright: ignore[reportMissingModuleSource]
        BOPAlgo_MakeConnected,
    )
    from OCP.BRep import BRep_Builder  # pyright: ignore[reportMissingModuleSource]
    from OCP.BRepBndLib import (  # pyright: ignore[reportMissingModuleSource]
        BRepBndLib,
    )
    from OCP.BRepTools import BRepTools  # pyright: ignore[reportMissingModuleSource]
    from OCP.IFSelect import (  # pyright: ignore[reportMissingModuleSource]
        IFSelect_RetDone,
    )
    from OCP.STEPControl import (  # pyright: ignore[reportMissingModuleSource]
        STEPControl_Reader,
    )
    from OCP.TopAbs import (  # pyright: ignore[reportMissingModuleSource]
        TopAbs_ShapeEnum,
        TopAbs_SOLID,
    )
    from OCP.TopExp import TopExp_Explorer  # pyright: ignore[reportMissingModuleSource]
    from OCP.TopoDS import (  # pyright: ignore[reportMissingModuleSource]
        TopoDS,
        TopoDS_Face,
        TopoDS_Shape,
        TopoDS_Shell,
        TopoDS_Solid,
    )
except ImportError as e:
    raise ImportError(
        "OCP not found. See Stellarmesh installation instructions."
    ) from e

logger = logging.getLogger(__name__)


class Face(Protocol):
    """Interface for a CadQuery or Build123d Face."""

    wrapped: TopoDS_Face | None


class Shell(Protocol):
    """Interface for a CadQuery or Build123d Shell."""

    wrapped: TopoDS_Shell | None


class Solid(Protocol):
    """Interface for a CadQuery or Build123d Solid."""

    wrapped: TopoDS_Solid | None


class Geometry:
    """Geometry, representing an ordered list of solids, to be meshed."""

    solids: list[TopoDS_Solid]
    material_names: list[str]
    faces: list[TopoDS_Face]
    face_boundary_conditions: list[str]

    def __init__(
        self,
        solids: Optional[Sequence[Solid | TopoDS_Solid]] = None,
        material_names: Optional[Sequence[str]] = None,
        surfaces: Optional[Sequence[Face | Shell | TopoDS_Face | TopoDS_Shell]] = None,
        surface_boundary_conditions: Optional[Sequence[str]] = None,
    ):
        """Construct geometry from solids.

        Args:
            solids: List of solids, where each solid is a build123d Solid, CadQuery
            Solid, or OCP TopoDS_Solid.
            material_names: List of materials. Must match length of solids.
            surfaces: List of surfaces, where each surface is a build123d or Cadquery
            Face or Shell, or an OCP TopoDS_Face or TopoDS_Shell.
            surface_boundary_conditions: List of boundary condition names. Must match
            length of surfaces.
        """
        if (solids and not material_names) or (material_names and not solids):
            raise ValueError(
                "If solids or material_names are provided"
                ", both must be provided and match in length."
            )

        if (surfaces and not surface_boundary_conditions) or (
            surface_boundary_conditions and not surfaces
        ):
            raise ValueError(
                "If surfaces or surface_boundary_conditions are provided"
                ", both must be provided and match in length."
            )

        self.solids = []
        self.material_names = []
        if solids and material_names:
            for i, (s, mat_name) in enumerate(zip(solids, material_names, strict=True)):
                s_wrapped = (
                    s if isinstance(s, TopoDS_Shape) else getattr(s, "wrapped", None)
                )

                if s_wrapped is None:
                    raise ValueError(
                        f"Solid {i} has no wrapped TopoDS_Shape. Is it valid?"
                    )
                elif s_wrapped.ShapeType() != TopAbs_SOLID:
                    raise ValueError(
                        f"Solid {i} is not of type TopABS_Solid but rather of type"
                        + str(s_wrapped.ShapeType().name)
                    )

                self.solids.append(s_wrapped)
                self.material_names.append(mat_name)

        self.faces = []
        self.face_boundary_conditions = []
        if surfaces and surface_boundary_conditions:
            for i, (s, bc) in enumerate(
                zip(surfaces, surface_boundary_conditions, strict=True)
            ):
                s_wrapped = (
                    s
                    if isinstance(s, (TopoDS_Face, TopoDS_Shell))
                    else getattr(s, "wrapped", None)
                )

                if s.wrapped is None:  # type: ignore
                    raise ValueError(
                        f"{s} {i} has no wrapped TopoDS_Shape. Is it valid?"
                    )

                if isinstance(s_wrapped, TopoDS_Face):
                    self.faces.append(s_wrapped)
                    self.face_boundary_conditions.append(bc)
                elif isinstance(s_wrapped, TopoDS_Shell):
                    child_faces = self._get_child_shapes(s_wrapped, TopoDS_Face)
                    self.faces.extend(child_faces)
                    self.face_boundary_conditions.extend([bc] * len(child_faces))

                else:
                    raise TypeError(
                        f"Surface {i} is of invalid type {type(s).__name__}"
                    )

    @staticmethod
    @overload
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Face]
    ) -> list[TopoDS_Face]: ...

    @staticmethod
    @overload
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Shell]
    ) -> list[TopoDS_Shell]: ...

    @staticmethod
    @overload
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Solid]
    ) -> list[TopoDS_Solid]: ...

    @staticmethod
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Shape]
    ) -> Sequence[TopoDS_Shape]:
        """Return all the child shapes of this shape."""
        type_map = {
            "TopoDS_Face": TopAbs_ShapeEnum.TopAbs_FACE,
            "TopoDS_Shell": TopAbs_ShapeEnum.TopAbs_SHELL,
            "TopoDS_Solid": TopAbs_ShapeEnum.TopAbs_SOLID,
        }

        top_abs_type = type_map.get(shape_type.__name__)
        if top_abs_type is None:
            raise ValueError(f"Unsupported shape_type: {shape_type}")

        shapes = []
        explorer = TopExp_Explorer(parent, top_abs_type)
        while explorer.More():
            assert explorer.Current().ShapeType() == top_abs_type
            shapes.append(explorer.Current())
            explorer.Next()
        return shapes

    @classmethod
    def _get_solids_from_shape(cls, shape: TopoDS_Shape) -> list[TopoDS_Solid]:
        """Return all the solids in this shape."""
        solids: list[TopoDS_Solid] = []
        if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID:
            solids.append(TopoDS.Solid_s(shape))
        elif shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            solids = cls._get_child_shapes(shape, TopoDS_Solid)
        return solids

    # TODO(akoen): from_step and from_brep are not DRY
    # https://github.com/Thea-Energy/stellarmesh/issues/2
    @classmethod
    def from_step(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a step file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        logger.info(f"Importing {filename}")

        reader = STEPControl_Reader()
        read_status = reader.ReadFile(filename)
        if read_status != IFSelect_RetDone:
            raise ValueError(f"STEP File {filename} could not be loaded")
        for i in range(reader.NbRootsForTransfer()):
            reader.TransferRoot(i + 1)

        solids = []
        for i in range(reader.NbShapes()):
            shape = reader.Shape(i + 1)
            solids.extend(cls._get_solids_from_shape(shape))

        return cls(solids, material_names)

    @classmethod
    def import_step(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a step file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        warnings.warn(
            "The import_step method is deprecated. Use from_step instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_step(filename, material_names)

    @classmethod
    def from_brep(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a brep (cadquery, build123d native) file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        logger.info(f"Importing {filename}")

        shape = TopoDS_Shape()
        builder = BRep_Builder()
        BRepTools.Read_s(shape, filename, builder)

        if shape.IsNull():
            raise ValueError(f"Could not import {filename}")
        solids = cls._get_solids_from_shape(shape)

        logger.info(f"Importing {len(solids)} from {filename}")
        return cls(solids, material_names)

    @classmethod
    def import_brep(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a brep (cadquery, build123d native) file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        warnings.warn(
            "The import_brep method is deprecated. Use from_brep instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_brep(filename, material_names)

    @staticmethod
    def _get_bounding_box(
        solid: TopoDS_Solid,
    ) -> tuple[float, float, float, float, float, float]:
        """Return (xmin, ymin, zmin, xmax, ymax, zmax) bounding box of a solid."""
        bbox = Bnd_Box()
        BRepBndLib.Add_s(solid, bbox)
        return bbox.Get()

    @staticmethod
    def _bounding_boxes_overlap(
        bb1: tuple[float, float, float, float, float, float],
        bb2: tuple[float, float, float, float, float, float],
    ) -> bool:
        """Check if two axis-aligned bounding boxes overlap."""
        xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = bb1
        xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = bb2
        return not (
            xmax2 < xmin1
            or xmax1 < xmin2
            or ymax2 < ymin1
            or ymax1 < ymin2
            or zmax2 < zmin1
            or zmax1 < zmin2
        )

    def _imprint_group(self, indices: list[int]) -> list[TopoDS_Solid]:
        """Imprint a group of solids by their indices.

        Args:
            indices: Indices into self.solids to imprint together.

        Returns:
            List of imprinted solids in the same order as indices.
        """
        if len(indices) == 1:
            return [self.solids[indices[0]]]

        bldr = BOPAlgo_MakeConnected()
        bldr.SetRunParallel(theFlag=True)
        bldr.SetUseOBB(theUseOBB=True)

        for idx in indices:
            bldr.AddArgument(self.solids[idx])

        bldr.Perform()
        res = bldr.Shape()
        res_solids = self._get_solids_from_shape(res)

        if len(res_solids) != len(indices):
            raise RuntimeError(
                f"Length of imprinted solids {len(res_solids)} "
                f"!= length of input solids {len(indices)}"
            )

        return res_solids

    def imprint(self, batch_size: Optional[int] = None) -> Geometry:
        """Imprint faces of current geometry.

        When batch_size is None (default), all solids are imprinted in a single
        operation. For large geometries this may require excessive memory.

        When batch_size is set, solids are processed in staged batches:
        1. Bounding boxes are computed for all solids.
        2. Solids are grouped into connected components based on AABB overlap.
        3. Each connected component is imprinted independently. If a component
           exceeds batch_size, it is further subdivided into batches that are
           imprinted iteratively.

        This staged approach reduces peak memory usage at the cost of additional
        computation, similar to the approach used in OpenMSR/CAD_to_OpenMC.

        Args:
            batch_size: Maximum number of solids to imprint in a single operation.
                If None, all solids are imprinted at once (original behavior).

        Returns:
            A new geometry with the imprinted and merged geometry.
        """
        if batch_size is not None and batch_size < 2:
            raise ValueError("batch_size must be at least 2 or None")

        if batch_size is None:
            return self._imprint_monolithic()

        return self._imprint_staged(batch_size)

    def _imprint_monolithic(self) -> Geometry:
        """Imprint all solids in a single operation (original behavior)."""
        bldr = BOPAlgo_MakeConnected()
        bldr.SetRunParallel(theFlag=True)
        bldr.SetUseOBB(theUseOBB=True)

        for solid in self.solids:
            bldr.AddArgument(solid)

        bldr.Perform()
        res = bldr.Shape()
        res_solids = self._get_solids_from_shape(res)

        if (l0 := len(res_solids)) != (l1 := len(self.solids)):
            raise RuntimeError(
                f"Length of imprinted solids {l0} != length of original solids {l1}"
            )

        return type(self)(res_solids, self.material_names)

    def _imprint_staged(self, batch_size: int) -> Geometry:
        """Imprint solids in stages to reduce peak memory usage.

        Groups solids by AABB overlap into connected components, then imprints
        each component. Components larger than batch_size are subdivided.
        """
        n = len(self.solids)
        if n <= batch_size:
            logger.info(
                f"Staged imprint: {n} solids fit in one batch "
                f"(batch_size={batch_size}), using monolithic imprint."
            )
            return self._imprint_monolithic()

        # Compute bounding boxes.
        logger.info(f"Staged imprint: computing bounding boxes for {n} solids.")
        bboxes = [self._get_bounding_box(s) for s in self.solids]

        # Build adjacency and find connected components.
        adjacency = self._build_adjacency(n, bboxes)
        components = self._find_connected_components(n, adjacency)

        logger.info(
            f"Staged imprint: found {len(components)} connected components "
            f"(sizes: {[len(c) for c in components]})."
        )

        # Process each component.
        result_solids: list[Optional[TopoDS_Solid]] = [None] * n
        for component in components:
            self._imprint_component(
                component, batch_size, adjacency, result_solids
            )

        assert all(s is not None for s in result_solids)
        return type(self)(
            list(result_solids),  # type: ignore[arg-type]
            self.material_names,
        )

    def _build_adjacency(
        self,
        n: int,
        bboxes: list[tuple[float, float, float, float, float, float]],
    ) -> list[set[int]]:
        """Build adjacency list based on AABB overlap."""
        adjacency: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._bounding_boxes_overlap(bboxes[i], bboxes[j]):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return adjacency

    @staticmethod
    def _find_connected_components(
        n: int, adjacency: list[set[int]]
    ) -> list[list[int]]:
        """Find connected components using BFS."""
        visited = [False] * n
        components: list[list[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            component: list[int] = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)
        return components

    def _imprint_component(
        self,
        component: list[int],
        batch_size: int,
        adjacency: list[set[int]],
        result_solids: list[Optional[TopoDS_Solid]],
    ) -> None:
        """Imprint a single connected component, batching if needed."""
        if len(component) == 1:
            result_solids[component[0]] = self.solids[component[0]]
        elif len(component) <= batch_size:
            logger.info(
                f"Staged imprint: imprinting component of "
                f"{len(component)} solids."
            )
            imprinted = self._imprint_group(component)
            for idx, solid in zip(component, imprinted, strict=True):
                result_solids[idx] = solid
        else:
            logger.info(
                f"Staged imprint: subdividing component of "
                f"{len(component)} solids into batches of {batch_size}."
            )
            self._imprint_large_component(
                component, batch_size, adjacency, result_solids
            )

    def _imprint_large_component(
        self,
        component: list[int],
        batch_size: int,
        adjacency: list[set[int]],
        result_solids: list[Optional[TopoDS_Solid]],
    ) -> None:
        """Imprint a large connected component in batches.

        Processes the component in sequential batches of batch_size.

        Args:
            component: Indices of solids in this connected component.
            batch_size: Maximum solids per batch.
            adjacency: Adjacency list for AABB overlap.
            result_solids: Output list to populate with imprinted solids.
        """
        # Sort component by adjacency degree (most connected first) to improve
        # the chance that overlapping solids end up in the same batch.
        sorted_component = sorted(
            component, key=lambda i: len(adjacency[i] & set(component)), reverse=True
        )

        # Process in sequential batches.
        processed = 0
        while processed < len(sorted_component):
            batch_indices = sorted_component[processed : processed + batch_size]
            logger.info(
                f"Staged imprint: processing batch of {len(batch_indices)} solids "
                f"({processed}/{len(sorted_component)} done)."
            )
            imprinted = self._imprint_group(batch_indices)
            for idx, solid in zip(batch_indices, imprinted, strict=True):
                result_solids[idx] = solid
                # Update self.solids so subsequent batches use imprinted versions.
                self.solids[idx] = solid
            processed += batch_size

"""Stellarmesh geometry.

name: geometry.py
author: Alex Koen

desc: Geometry class represents a CAD geometry to be meshed.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import deque
from typing import (
    Callable,
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
    from OCP.STEPCAFControl import (  # pyright: ignore[reportMissingModuleSource]
        STEPCAFControl_Reader,
    )
    from OCP.STEPControl import (  # pyright: ignore[reportMissingModuleSource]
        STEPControl_Reader,
    )
    from OCP.TCollection import (  # pyright: ignore[reportMissingModuleSource]
        TCollection_ExtendedString,
    )
    from OCP.TDataStd import TDataStd_Name  # pyright: ignore[reportMissingModuleSource]
    from OCP.TDF import (  # pyright: ignore[reportMissingModuleSource]
        TDF_Label,
        TDF_LabelSequence,
    )
    from OCP.TDocStd import (
        TDocStd_Document,  # pyright: ignore[reportMissingModuleSource]
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
    from OCP.XCAFApp import (  # pyright: ignore[reportMissingModuleSource]
        XCAFApp_Application,
    )
    from OCP.XCAFDoc import (  # pyright: ignore[reportMissingModuleSource]
        XCAFDoc_DocumentTool,
        XCAFDoc_ShapeTool,
    )
except ImportError as e:
    raise ImportError(
        "OCP not found. See Stellarmesh installation instructions."
    ) from e

logger = logging.getLogger(__name__)


def _get_ocp_method(obj: object, name: str) -> Callable[..., object]:
    """Return the OCP method, supporting both suffixed and unsuffixed names."""
    method = getattr(obj, f"{name}_s", None) or getattr(obj, name, None)
    if method is None:
        raise AttributeError(f"{obj!r} has no {name} or {name}_s method")
    return method


def _ocp_string_to_str(value: object) -> str:
    """Convert OCP string wrapper objects to Python strings."""
    if isinstance(value, str):
        return value

    to_ext_string = getattr(value, "ToExtString", None)
    if callable(to_ext_string):
        return to_ext_string()

    return str(value)


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
    assembly_names: list[list[str]]
    faces: list[TopoDS_Face]
    face_boundary_conditions: list[str]

    def __init__(
        self,
        solids: Optional[Sequence[Solid | TopoDS_Solid]] = None,
        material_names: Optional[Sequence[str]] = None,
        surfaces: Optional[Sequence[Face | Shell | TopoDS_Face | TopoDS_Shell]] = None,
        surface_boundary_conditions: Optional[Sequence[str]] = None,
        assembly_names: Optional[Sequence[Sequence[str]]] = None,
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
            assembly_names: Assembly names containing each solid. Must match the
                length of solids. Each solid may belong to multiple nested assemblies.
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
        self.assembly_names = []
        if solids and material_names:
            if assembly_names is not None and len(assembly_names) != len(solids):
                raise ValueError(
                    f"Length of assembly_names ({len(assembly_names)}) does not match "
                    f"number of solids ({len(solids)})."
                )
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
                names = assembly_names[i] if assembly_names is not None else ()
                if isinstance(names, str):
                    raise TypeError(
                        "Each assembly_names entry must be a sequence of names, "
                        "not a string."
                    )
                self.assembly_names.append(list(dict.fromkeys(names)))

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

    def get_material_names(self) -> list[str]:
        """Return a copy of the current material names.

        Returns:
            List of material name strings, one per solid.
        """
        return list(self.material_names)

    def set_material_names(self, material_names: Sequence[str]) -> None:
        """Set new material names for the geometry.

        Args:
            material_names: List of material names. Must match the number of solids.

        Raises:
            ValueError: If the length of material_names does not match the number of
                solids.
        """
        if len(material_names) != len(self.solids):
            raise ValueError(
                f"Length of material_names ({len(material_names)}) does not match "
                f"number of solids ({len(self.solids)})."
            )
        self.material_names = list(material_names)

    def get_assembly_names(self) -> list[list[str]]:
        """Return assembly memberships for each solid."""
        return [list(names) for names in self.assembly_names]

    def set_assembly_names(self, assembly_names: Sequence[Sequence[str]]) -> None:
        """Set assembly memberships for each solid."""
        if len(assembly_names) != len(self.solids):
            raise ValueError(
                f"Length of assembly_names ({len(assembly_names)}) does not match "
                f"number of solids ({len(self.solids)})."
            )
        normalized_names = []
        for names in assembly_names:
            if isinstance(names, str):
                raise TypeError(
                    "Each assembly_names entry must be a sequence of names, "
                    "not a string."
                )
            normalized_names.append(list(dict.fromkeys(names)))
        self.assembly_names = normalized_names

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
            solids.append(_get_ocp_method(TopoDS, "Solid")(shape))
        elif shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            solids = cls._get_child_shapes(shape, TopoDS_Solid)
        return solids

    @classmethod
    def _extract_step_metadata(
        cls, filename: str, n_solids: int
    ) -> tuple[list[str], list[list[str]]]:
        """Extract part names and assembly memberships using the XDE framework.

        Reads the STEP file with XDE (Extended Data Framework) to access the
        assembly structure and extract the name associated with each solid.

        Args:
            filename: Path to the STEP file.
            n_solids: Expected number of solids (for validation).

        Returns:
            Part names and assembly memberships, one entry per solid.
        """
        app = _get_ocp_method(XCAFApp_Application, "GetApplication")()
        # "XmlOcaf" is the standard XDE document format for OCAF applications.
        doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        app.InitDocument(doc)

        caf_reader = STEPCAFControl_Reader()
        caf_reader.SetNameMode(True)  # noqa: FBT003
        read_status = caf_reader.ReadFile(filename)
        if read_status != IFSelect_RetDone:
            raise ValueError(f"STEP File {filename} could not be loaded")
        caf_reader.Transfer(doc)

        _get_shape_tool = (
            getattr(XCAFDoc_DocumentTool, "ShapeTool_s", None)
            or getattr(XCAFDoc_DocumentTool, "ShapeTool", None)
            or getattr(XCAFDoc_ShapeTool, "GetTool_s", None)
            or getattr(XCAFDoc_ShapeTool, "GetTool", None)
        )
        if _get_shape_tool is None:
            raise AttributeError(
                "No compatible XCAF shape tool accessor found in this OCP build"
            )
        shape_tool = _get_shape_tool(doc.Main())

        labels = TDF_LabelSequence()
        shape_tool.GetFreeShapes(labels)
        names: list[str] = []
        assembly_names: list[list[str]] = []

        def _get_label_name(label: TDF_Label) -> str:
            """Get the name attribute from a label, or empty string."""
            name_attr = TDataStd_Name()
            if label.FindAttribute(
                _get_ocp_method(TDataStd_Name, "GetID")(), name_attr
            ):
                name_str = name_attr.Get()
                return _ocp_string_to_str(name_str) if name_str else ""
            return ""

        def _normalize_assembly_name(name: str) -> str:
            """Remove Onshape's per-instance suffix, e.g. `` <4>``."""
            return re.sub(r"\s*<\d+>$", "", name).strip()

        def _collect_names_from_label(
            label: TDF_Label,
            parent_assemblies: Sequence[str],
            instance_name: str = "",
        ) -> None:
            """Recursively collect solid metadata from the assembly tree."""
            shape = _get_ocp_method(shape_tool, "GetShape")(label)
            if shape is None or shape.IsNull():
                return

            assemblies = list(parent_assemblies)
            if _get_ocp_method(shape_tool, "IsAssembly")(label):
                assembly_name = _normalize_assembly_name(
                    instance_name or _get_label_name(label)
                )
                if assembly_name and assembly_name not in assemblies:
                    assemblies.append(assembly_name)

            if shape.ShapeType() == TopAbs_SOLID:
                names.append(_get_label_name(label) or instance_name)
                assembly_names.append(assemblies)
            elif shape.ShapeType() in (
                TopAbs_ShapeEnum.TopAbs_COMPOUND,
                TopAbs_ShapeEnum.TopAbs_COMPSOLID,
            ):
                # Recurse into sub-labels (components of assembly)
                sub_labels = TDF_LabelSequence()
                _get_ocp_method(shape_tool, "GetComponents")(label, sub_labels)
                for j in range(sub_labels.Length()):
                    sub_label = sub_labels.Value(j + 1)
                    ref_label = TDF_Label()
                    if _get_ocp_method(shape_tool, "GetReferredShape")(
                        sub_label, ref_label
                    ):
                        _collect_names_from_label(
                            ref_label,
                            assemblies,
                            _get_label_name(sub_label),
                        )
                    else:
                        _collect_names_from_label(sub_label, assemblies)

        for i in range(labels.Length()):
            label = labels.Value(i + 1)
            _collect_names_from_label(label, ())

        if len(names) != n_solids:
            logger.warning(
                f"Number of extracted part names ({len(names)}) does not match "
                f"number of solids ({n_solids}). Using generic names."
            )
            names = [f"solid_{i}" for i in range(n_solids)]
            assembly_names = [[] for _ in range(n_solids)]

        logger.info(f"Extracted material names from STEP file: {names}")
        logger.info(f"Extracted assembly names from STEP file: {assembly_names}")
        return names, assembly_names

    @classmethod
    def _extract_step_part_names(cls, filename: str, n_solids: int) -> list[str]:
        """Extract part names from a STEP file using the XDE framework."""
        return cls._extract_step_metadata(filename, n_solids)[0]

    # TODO(akoen): from_step and from_brep are not DRY
    # https://github.com/Thea-Energy/stellarmesh/issues/2
    @classmethod
    def from_step(
        cls,
        filename: str,
        material_names: Optional[Sequence[str]] = None,
        assembly_names: Optional[Sequence[Sequence[str]]] = None,
    ) -> Geometry:
        """Import model from a step file.

        If material_names is not provided, names are automatically extracted from
        the assembly part names in the STEP file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.
                If None, material names are extracted from STEP part names.
            assembly_names: Assembly memberships matching solids in file. If None,
                names are extracted from the STEP assembly hierarchy.

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

        if material_names is None or assembly_names is None:
            extracted_materials, extracted_assemblies = cls._extract_step_metadata(
                filename, len(solids)
            )
            if material_names is None:
                material_names = extracted_materials
            if assembly_names is None:
                assembly_names = extracted_assemblies

        return cls(solids, material_names, assembly_names=assembly_names)

    @classmethod
    def import_step(
        cls,
        filename: str,
        material_names: Optional[Sequence[str]] = None,
        assembly_names: Optional[Sequence[Sequence[str]]] = None,
    ) -> Geometry:
        """Import model from a step file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.
            assembly_names: Assembly memberships matching solids in file.

        Returns:
            Model.
        """
        warnings.warn(
            "The import_step method is deprecated. Use from_step instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_step(filename, material_names, assembly_names)

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
        return self._imprint_group_from(indices, self.solids)

    @staticmethod
    def _imprint_group_from(
        indices: list[int], solids: list[TopoDS_Solid]
    ) -> list[TopoDS_Solid]:
        """Imprint a group of solids from a given solids list.

        Args:
            indices: Indices into solids to imprint together.
            solids: The list of solids to index into.

        Returns:
            List of imprinted solids in the same order as indices.
        """
        if len(indices) == 1:
            return [solids[indices[0]]]

        bldr = BOPAlgo_MakeConnected()
        bldr.SetRunParallel(theFlag=True)
        bldr.SetUseOBB(theUseOBB=True)

        for idx in indices:
            bldr.AddArgument(solids[idx])

        bldr.Perform()
        res = bldr.Shape()
        res_solids = Geometry._get_solids_from_shape(res)

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

        return type(self)(
            res_solids,
            self.material_names,
            assembly_names=self.assembly_names,
        )

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
        # Work on a copy to avoid mutating self.solids during batch processing.
        working_solids = list(self.solids)
        for component in components:
            self._imprint_component(
                component, batch_size, adjacency, result_solids, working_solids
            )

        if not all(s is not None for s in result_solids):
            raise RuntimeError("Staged imprint failed: not all solids were processed.")
        return type(self)(
            list(result_solids),  # type: ignore[arg-type]
            self.material_names,
            assembly_names=self.assembly_names,
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
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
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
        working_solids: list[TopoDS_Solid],
    ) -> None:
        """Imprint a single connected component, batching if needed."""
        if len(component) == 1:
            result_solids[component[0]] = working_solids[component[0]]
        elif len(component) <= batch_size:
            logger.info(
                f"Staged imprint: imprinting component of {len(component)} solids."
            )
            imprinted = self._imprint_group_from(component, working_solids)
            for idx, solid in zip(component, imprinted, strict=True):
                result_solids[idx] = solid
        else:
            logger.info(
                f"Staged imprint: subdividing component of "
                f"{len(component)} solids into batches of {batch_size}."
            )
            self._imprint_large_component(
                component, batch_size, adjacency, result_solids, working_solids
            )

    def _imprint_large_component(
        self,
        component: list[int],
        batch_size: int,
        adjacency: list[set[int]],
        result_solids: list[Optional[TopoDS_Solid]],
        working_solids: list[TopoDS_Solid],
    ) -> None:
        """Imprint a large connected component in batches.

        Processes the component in sequential batches of batch_size.

        Args:
            component: Indices of solids in this connected component.
            batch_size: Maximum solids per batch.
            adjacency: Adjacency list for AABB overlap.
            result_solids: Output list to populate with imprinted solids.
            working_solids: Mutable working copy of solids list.
        """
        # Sort component by adjacency degree (most connected first) to improve
        # the chance that overlapping solids end up in the same batch.
        component_set = set(component)
        sorted_component = sorted(
            component, key=lambda i: len(adjacency[i] & component_set), reverse=True
        )

        # Process in sequential batches.
        processed = 0
        while processed < len(sorted_component):
            batch_indices = sorted_component[processed : processed + batch_size]
            logger.info(
                f"Staged imprint: processing batch of {len(batch_indices)} solids "
                f"({processed}/{len(sorted_component)} done)."
            )
            imprinted = self._imprint_group_from(batch_indices, working_solids)
            for idx, solid in zip(batch_indices, imprinted, strict=True):
                result_solids[idx] = solid
                # Update working_solids so subsequent batches use imprinted versions.
                working_solids[idx] = solid
            processed += batch_size

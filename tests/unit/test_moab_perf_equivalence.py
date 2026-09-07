import build123d as bd
import numpy as np
import pymoab.core
import pymoab.types
import pytest
import stellarmesh as sm
from pymoab.rng import Range
from stellarmesh import moab as sm_moab


def _build_imprinted_surface_mesh() -> sm.SurfaceMesh:
    box1 = bd.Solid.make_box(10.0, 10.0, 10.0)
    box2 = box1.transformed(offset=(0.0, 5.0, 10.0))
    geometry = sm.Geometry(
        [box1, box2],
        ["fe", "ss"],
        part_names=["fe - Block", "ss - Tank"],
        assembly_names=[["assembly"], ["assembly", "subassembly"]],
    ).imprint()
    mesh = sm.SurfaceMesh.from_geometry(
        geometry, sm.GmshSurfaceOptions(max_mesh_size=5)
    )
    with mesh:
        surface_tag = sm_moab.gmsh.model.get_entities(2)[0][1]
        mesh.entity_metadata(2, surface_tag).boundary_condition = "vacuum"
    return mesh


def _reference_add_nodes(model: sm.DAGMCModel) -> dict[int, int]:
    node_tags, coords, _ = sm_moab.gmsh.model.mesh.get_nodes()
    if np.isnan(coords).any():
        raise ValueError("Mesh coordinates contain NaNs.")
    if np.isinf(coords).any():
        raise ValueError("Mesh coordinates contain infinite values.")

    moab_vertices = model._core.create_vertices(coords)
    model._core.tag_set_data(model.id_tag, moab_vertices, node_tags.astype(np.int32))  # pyright: ignore[reportAttributeAccessIssue]

    node_tag_map = dict(zip(node_tags, moab_vertices, strict=True))
    if len(node_tag_map) != len(node_tags):
        raise ValueError("Duplicate node tags found.")
    return node_tag_map


def _reference_create_elements(
    model: sm.DAGMCModel, dim: int, tag: int, node_tag_map: dict[int, int]
) -> Range:
    element_types, _, node_tags_list = sm_moab.gmsh.model.mesh.get_elements(dim, tag)
    all_new_handles = []

    for elem_type, node_tags in zip(element_types, node_tags_list, strict=True):
        if elem_type == 2:
            moab_type, nodes_per_elem = pymoab.types.MBTRI, 3
        elif elem_type == 4:
            moab_type, nodes_per_elem = pymoab.types.MBTET, 4
        elif elem_type == 5:
            moab_type, nodes_per_elem = pymoab.types.MBHEX, 8
        else:
            continue

        conn = np.array([node_tag_map[t] for t in node_tags], dtype=np.uint64).reshape(
            -1, nodes_per_elem
        )
        all_new_handles.extend(model._core.create_element(moab_type, c) for c in conn)

    return Range(all_new_handles)


def _reference_create_surface_elements(
    model: sm.DAGMCModel,
    surface_tag: int,
    surface_set: sm.DAGMCSurface,
    node_tag_map: dict[int, int],
):
    triangles = _reference_create_elements(model, 2, surface_tag, node_tag_map)
    if not triangles:
        raise RuntimeError(f"Surface {surface_tag} has no elements")

    model._core.add_entities(surface_set.handle, triangles)
    adj_verts = model._core.get_adjacencies(triangles, 0, create_if_missing=False)
    model._core.add_entities(surface_set.handle, adj_verts)


def _reference_add_surfaces(
    model: sm.DAGMCModel, mesh: sm.Mesh, node_tag_map: dict[int, int]
) -> dict[int, sm.DAGMCSurface]:
    surface_map: dict[int, sm.DAGMCSurface] = {}
    surface_tags = [tag for _, tag in sm_moab.gmsh.model.get_entities(2)]

    for surface_tag in surface_tags:
        surface_set = model.create_surface(surface_tag)
        surface_map[surface_tag] = surface_set

        if (bc := mesh.entity_metadata(2, surface_tag).boundary_condition) is not None:
            surface_set.boundary = bc

        _reference_create_surface_elements(
            model, surface_tag, surface_set, node_tag_map
        )
        model._create_volume_friend_for_lonely_surfaces(surface_tag, surface_set)

    return surface_map


def _reference_add_volumes(
    model: sm.DAGMCModel, mesh: sm.Mesh, surface_map: dict[int, sm.DAGMCSurface]
):
    volume_tags = [tag for _, tag in sm_moab.gmsh.model.get_entities(3)]
    volume_map: dict[int, sm.DAGMCVolume] = {}

    for volume_tag in volume_tags:
        volume_set = model.create_volume(volume_tag)
        volume_map[volume_tag] = volume_set
        metadata = mesh.entity_metadata(3, volume_tag)
        volume_set.material = metadata.material
        if (part_name := metadata.part) is not None:
            volume_set.part = part_name

    next_group_id = max((g.global_id for g in model.groups), default=0) + 1
    for dim, physical_tag in sm_moab.gmsh.model.get_physical_groups(3):
        group_name = sm_moab.gmsh.model.get_physical_name(dim, physical_tag)
        if not group_name.startswith("assembly:"):
            continue
        group = model.create_group(group_name)
        group.global_id = next_group_id
        next_group_id += 1
        for volume_tag in sm_moab.gmsh.model.get_entities_for_physical_group(
            dim, physical_tag
        ):
            group.add(volume_map[volume_tag])

    for surface_tag, surface in surface_map.items():
        metadata = mesh.entity_metadata(2, surface_tag)
        if (forward_vol_tag := metadata.forward_volume) is not None:
            surface.forward_volume = volume_map[forward_vol_tag]
        if (reverse_vol_tag := metadata.reverse_volume) is not None:
            surface.reverse_volume = volume_map[reverse_vol_tag]

    for volume in volume_map.values():
        if not model._core.get_child_meshsets(volume.handle):
            sm_moab.logger.error(f"Volume {volume.global_id} has no assigned surfaces.")


def _build_reference_dagmc(mesh: sm.Mesh) -> sm.DAGMCModel:
    model = sm.DAGMCModel(pymoab.core.Core())
    with mesh:
        sm_moab.gmsh.model.mesh.removeDuplicateNodes()
        node_tag_map = _reference_add_nodes(model)
        surface_map = _reference_add_surfaces(model, mesh, node_tag_map)
        _reference_add_volumes(model, mesh, surface_map)
        model._finalize_file_set()
    return model


def _vertex_count(model: sm.DAGMCModel) -> int:
    return len(
        model._core.get_entities_by_type(
            model.root_set, pymoab.types.MBVERTEX, recur=True
        )
    )


def _summary_counts(model: sm.DAGMCModel) -> tuple[int, int, int, int, int]:
    return (
        _vertex_count(model),
        len(model.triangles),
        len(model.surfaces),
        len(model.volumes),
        len(model.groups),
    )


def _group_members_by_name(
    model: sm.DAGMCModel,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    return {
        group.name: (
            tuple(sorted(volume.global_id for volume in group.volumes)),
            tuple(sorted(surface.global_id for surface in group.surfaces)),
        )
        for group in model.groups
    }


def _vertex_coordinates(model: sm.DAGMCModel) -> set[tuple[float, float, float]]:
    vertices = model._core.get_entities_by_type(
        model.root_set, pymoab.types.MBVERTEX, recur=True
    )
    return {
        tuple(np.round(model._core.get_coords(np.array([vertex])).reshape(3), 12))
        for vertex in vertices
    }


def _triangle_connectivity(model: sm.DAGMCModel) -> set[tuple[int, int, int]]:
    connectivity: set[tuple[int, int, int]] = set()
    for triangle in model.triangles:
        node_ids = model._core.tag_get_data(
            model.id_tag, model._core.get_connectivity(triangle), flat=True
        )
        sorted_node_ids = sorted(int(node_id) for node_id in node_ids)
        tri = (sorted_node_ids[0], sorted_node_ids[1], sorted_node_ids[2])
        connectivity.add(tri)
    return connectivity


def _surface_by_global_id(model: sm.DAGMCModel) -> dict[int, sm.DAGMCSurface]:
    return {surface.global_id: surface for surface in model.surfaces}


def test_dagmc_model_batched_builder_matches_reference(tmp_path):
    mesh = _build_imprinted_surface_mesh()
    optimized = sm.DAGMCModel.from_mesh(mesh)
    reference = _build_reference_dagmc(mesh)

    assert _summary_counts(optimized) == _summary_counts(reference)
    assert _group_members_by_name(optimized) == _group_members_by_name(reference)

    optimized_surfaces = _surface_by_global_id(optimized)
    reference_surfaces = _surface_by_global_id(reference)
    assert set(optimized_surfaces) == set(reference_surfaces)
    for global_id in optimized_surfaces:
        opt_surface = optimized_surfaces[global_id]
        ref_surface = reference_surfaces[global_id]
        assert len(opt_surface.triangles) == len(ref_surface.triangles)
        assert sorted(vol.global_id for vol in opt_surface.adjacent_volumes) == sorted(
            vol.global_id for vol in ref_surface.adjacent_volumes
        )
        assert (
            opt_surface.forward_volume.global_id if opt_surface.forward_volume else None
        ) == (
            ref_surface.forward_volume.global_id if ref_surface.forward_volume else None
        )
        assert (
            opt_surface.reverse_volume.global_id if opt_surface.reverse_volume else None
        ) == (
            ref_surface.reverse_volume.global_id if ref_surface.reverse_volume else None
        )

    assert _vertex_coordinates(optimized) == _vertex_coordinates(reference)
    assert _triangle_connectivity(optimized) == _triangle_connectivity(reference)
    assert optimized.material_to_volume_ids == reference.material_to_volume_ids
    assert optimized.part_to_volume_ids == reference.part_to_volume_ids
    assert optimized.assembly_to_volume_ids == reference.assembly_to_volume_ids

    optimized_path = tmp_path / "optimized.h5m"
    reference_path = tmp_path / "reference.h5m"
    optimized.write(optimized_path)
    reference.write(reference_path)
    reloaded_optimized = sm.DAGMCModel(optimized_path)
    reloaded_reference = sm.DAGMCModel(reference_path)
    assert _summary_counts(reloaded_optimized) == _summary_counts(reloaded_reference)


def test_create_elements_handles_multiple_element_types(monkeypatch):
    model = sm.MOABModel(pymoab.core.Core())
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    vertices = model._core.create_vertices(coords.reshape(-1))
    vertex_handles = np.fromiter(vertices, dtype=np.uint64, count=vertices.size())
    node_lookup = np.zeros(5, dtype=np.uint64)
    node_lookup[1:] = vertex_handles

    def _mock_get_elements(_dim, _tag):
        return (
            np.array([2, 4], dtype=np.int32),
            [np.array([1], dtype=np.uint64), np.array([2], dtype=np.uint64)],
            [
                np.array([1, 2, 3], dtype=np.uint64),
                np.array([1, 2, 3, 4], dtype=np.uint64),
            ],
        )

    monkeypatch.setattr(sm_moab.gmsh.model.mesh, "get_elements", _mock_get_elements)
    created = model._create_elements(3, 1, node_lookup)

    assert len(created) == 2
    assert (
        len(model._core.get_entities_by_type(model.root_set, pymoab.types.MBTRI)) == 1
    )
    assert (
        len(model._core.get_entities_by_type(model.root_set, pymoab.types.MBTET)) == 1
    )


def test_create_elements_raises_on_unknown_node_tag(monkeypatch):
    model = sm.MOABModel(pymoab.core.Core())
    node_lookup = np.zeros(5, dtype=np.uint64)

    def _mock_get_elements(_dim, _tag):
        return (
            np.array([2], dtype=np.int32),
            [np.array([1], dtype=np.uint64)],
            [np.array([1, 2, 9], dtype=np.uint64)],
        )

    monkeypatch.setattr(sm_moab.gmsh.model.mesh, "get_elements", _mock_get_elements)

    with pytest.raises(ValueError, match="Encountered unknown node tag"):
        model._create_elements(2, 1, node_lookup)

import build123d as bd
import pymoab.core
import pymoab.types
import pytest
import stellarmesh as sm
from pymoab.rng import Range


@pytest.fixture(scope="module")
def dagmc_model():
    solid1 = bd.Solid.make_sphere(10.0)
    solid2 = bd.thicken(solid1.faces()[0], 10.0).solid()
    geom = sm.Geometry([solid1, solid2], ["iron", "iron"])
    mesh = sm.SurfaceMesh.from_geometry(geom, sm.GmshSurfaceOptions(max_mesh_size=5))
    return sm.DAGMCModel.from_mesh(mesh)


@pytest.fixture(scope="module")
def dagmc_model_named_parts():
    box1 = bd.Solid.make_box(10.0, 10.0, 10.0)
    box2 = box1.transformed(offset=(0.0, 5.0, 10.0))
    geom = sm.Geometry(
        [box1, box2],
        ["fe", "ss"],
        assembly_names=[["assembly"], ["assembly", "subassembly"]],
        part_names=["fe - Block", "ss - Tank"],
    )
    mesh = sm.SurfaceMesh.from_geometry(geom, sm.GmshSurfaceOptions(max_mesh_size=5))
    return sm.DAGMCModel.from_mesh(mesh)


class TestDAGMCModel:
    def test_surfaces(self, dagmc_model):
        assert isinstance(dagmc_model.surfaces, list)
        assert len(dagmc_model.surfaces) == 2
        assert isinstance(dagmc_model.surfaces[0], sm.DAGMCSurface)

    def test_volumes(self, dagmc_model):
        assert isinstance(dagmc_model.volumes, list)
        assert len(dagmc_model.volumes) == 2
        assert isinstance(dagmc_model.volumes[0], sm.DAGMCVolume)

    def test_global_id(self, dagmc_model):
        assert dagmc_model.surfaces[0].global_id == 1
        assert dagmc_model.volumes[0].global_id == 1

    def test_adjacent_surfaces(self, dagmc_model):
        vol = dagmc_model.volumes[0]
        surfaces = vol.adjacent_surfaces
        assert len(surfaces) == 1
        assert surfaces == [dagmc_model.surfaces[0]]

    def test_adjacent_volumes(self, dagmc_model):
        surf = dagmc_model.surfaces[0]
        volumes = surf.adjacent_volumes
        assert len(volumes) == 2
        assert volumes == [dagmc_model.volumes[0], dagmc_model.volumes[1]]
        assert surf.forward_volume == dagmc_model.volumes[0]
        assert surf.reverse_volume == dagmc_model.volumes[1]

    def test_tets(self, dagmc_model):
        assert dagmc_model.tets.empty()

    def test_triangles(self, dagmc_model):
        all_tris = dagmc_model.triangles
        assert isinstance(all_tris, Range)
        surf_tris = dagmc_model.surfaces[0].triangles
        assert isinstance(surf_tris, Range)
        assert all_tris.contains(surf_tris)

    def test_material(self, dagmc_model):
        vol = dagmc_model.volumes[0]
        assert vol.material == "iron"
        assert "mat:iron" in {group.name for group in vol.groups}

        vol.material = "plastic"
        assert vol.material == "plastic"
        vol_group_names = {group.name for group in vol.groups}
        assert "mat:iron" not in vol_group_names
        assert "mat:plastic" in vol_group_names

        all_group_names = {group.name for group in dagmc_model.groups}
        assert "mat:plastic" in all_group_names

    def test_group(self, dagmc_model):
        vol = dagmc_model.volumes[0]
        surf = dagmc_model.surfaces[0]

        group = dagmc_model.create_group("test_group")
        assert group.name == "test_group"

        group.name = "funny group"
        assert group.name == "funny group"

        group.add(vol)
        assert vol in group
        assert group.volumes == [vol]

        group.remove(vol)
        assert vol not in group
        assert group.volumes == []

        group.add(surf)
        assert group.surfaces == [surf]
        group.remove(surf)
        assert group.surfaces == []

    def test_long_group_name_round_trip(self, tmp_path):
        solid = bd.Solid.make_box(1.0, 1.0, 1.0)
        geometry = sm.Geometry([solid], ["steel"])
        mesh = sm.SurfaceMesh.from_geometry(
            geometry, sm.GmshSurfaceOptions(max_mesh_size=1.0)
        )
        dagmc_model = sm.DAGMCModel.from_mesh(mesh)
        long_name = "assembly:simplified storage tank - blanket salt"
        group = dagmc_model.create_group(long_name)
        group.global_id = 1
        group.add(dagmc_model.volumes[0])

        path = tmp_path / "long_name.h5m"
        dagmc_model.write(path)
        reloaded = sm.DAGMCModel(path)

        assert long_name in {group.name for group in reloaded.groups}
        # DAGMC hard-codes the NAME tag width, so it must not be widened.
        assert reloaded.name_tag.get_length() == pymoab.types.NAME_TAG_SIZE

    def test_long_name_keeps_file_dagmc_readable(self, tmp_path):
        solid = bd.Solid.make_box(1.0, 1.0, 1.0)
        geometry = sm.Geometry([solid], ["steel"])
        mesh = sm.SurfaceMesh.from_geometry(
            geometry, sm.GmshSurfaceOptions(max_mesh_size=1.0)
        )
        dagmc_model = sm.DAGMCModel.from_mesh(mesh)
        group = dagmc_model.create_group(
            "part:simplified storage tank - blanket salt"
        )
        group.global_id = 1
        group.add(dagmc_model.volumes[0])

        path = tmp_path / "dagmc_readable.h5m"
        dagmc_model.write(path)

        # Mimic DagMC, which pre-creates the NAME tag before loading the file.
        core = pymoab.core.Core()
        core.tag_get_handle(
            pymoab.types.NAME_TAG_NAME,
            pymoab.types.NAME_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )
        core.load_file(str(path))

    def test_long_material_name_raises(self, tmp_path):
        solid = bd.Solid.make_box(1.0, 1.0, 1.0)
        geometry = sm.Geometry([solid], ["steel"])
        mesh = sm.SurfaceMesh.from_geometry(
            geometry, sm.GmshSurfaceOptions(max_mesh_size=1.0)
        )
        dagmc_model = sm.DAGMCModel.from_mesh(mesh)
        with pytest.raises(ValueError, match="MOAB NAME tag"):
            dagmc_model.create_group("mat:" + "a" * 40)

    def test_repr(self, dagmc_model):
        surf = dagmc_model.surfaces[0]
        repr(surf)

        vol = dagmc_model.volumes[0]
        repr(vol)

        group = dagmc_model.groups[0]
        repr(group)

    def test_hash(self, dagmc_model):
        objects = {dagmc_model.surfaces[0], dagmc_model.volumes[0]}
        assert len(objects) == 2

    def test_material_to_volume_ids(self, dagmc_model_named_parts):
        mapping = dagmc_model_named_parts.material_to_volume_ids
        assert len(mapping["fe"]) == 1
        assert len(mapping["ss"]) == 1
        assert set(mapping["fe"]).isdisjoint(mapping["ss"])
        assert set(mapping["fe"] + mapping["ss"]) == {
            vol.global_id for vol in dagmc_model_named_parts.volumes
        }

    def test_part_to_volume_ids(self, dagmc_model_named_parts):
        mapping = dagmc_model_named_parts.part_to_volume_ids
        assert len(mapping["fe - Block"]) == 1
        assert len(mapping["ss - Tank"]) == 1
        assert dagmc_model_named_parts.volume_ids("fe - Block") == mapping["fe - Block"]
        assert (
            dagmc_model_named_parts.name_to_volume_ids["ss - Tank"]
            == mapping["ss - Tank"]
        )

    def test_assembly_to_volume_ids(self, dagmc_model_named_parts):
        mapping = dagmc_model_named_parts.assembly_to_volume_ids
        assert set(mapping["assembly"]) == {
            vol.global_id for vol in dagmc_model_named_parts.volumes
        }
        assert mapping["subassembly"] == [
            dagmc_model_named_parts.part_to_volume_ids["ss - Tank"][0]
        ]
        assert (
            dagmc_model_named_parts.name_to_volume_ids["assembly"]
            == mapping["assembly"]
        )

    def test_volume_bounding_box(self, dagmc_model_named_parts):
        volumes_by_id = {v.global_id: v for v in dagmc_model_named_parts.volumes}
        part_a_id = dagmc_model_named_parts.part_to_volume_ids["fe - Block"][0]
        part_b_id = dagmc_model_named_parts.part_to_volume_ids["ss - Tank"][0]
        vol_a_bbox = volumes_by_id[part_a_id].bounding_box
        vol_b_bbox = volumes_by_id[part_b_id].bounding_box
        assert vol_a_bbox[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-8)
        assert vol_a_bbox[1] == pytest.approx((10.0, 10.0, 10.0), abs=1e-8)
        assert vol_b_bbox[0] == pytest.approx((0.0, 5.0, 10.0), abs=1e-8)
        assert vol_b_bbox[1] == pytest.approx((10.0, 15.0, 20.0), abs=1e-8)

    def test_model_bounding_box_by_name_and_ids(self, dagmc_model_named_parts):
        bbox_from_name = dagmc_model_named_parts.bounding_box("ss - Tank")
        bbox_from_assembly = dagmc_model_named_parts.bounding_box("assembly")
        bbox_from_ids = dagmc_model_named_parts.bounding_box(
            dagmc_model_named_parts.part_to_volume_ids["fe - Block"]
            + dagmc_model_named_parts.part_to_volume_ids["ss - Tank"]
        )
        assert bbox_from_name[0] == pytest.approx((0.0, 5.0, 10.0), abs=1e-8)
        assert bbox_from_name[1] == pytest.approx((10.0, 15.0, 20.0), abs=1e-8)
        assert bbox_from_assembly[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-8)
        assert bbox_from_assembly[1] == pytest.approx((10.0, 15.0, 20.0), abs=1e-8)
        assert bbox_from_ids[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-8)
        assert bbox_from_ids[1] == pytest.approx((10.0, 15.0, 20.0), abs=1e-8)

    def test_model_bounding_box_errors(self, dagmc_model_named_parts):
        with pytest.raises(KeyError):
            dagmc_model_named_parts.bounding_box("missing_part")

        with pytest.raises(ValueError):
            dagmc_model_named_parts.bounding_box([999])


class TestMOABModel:
    def test_moabmodel_from_h5m(self, geom_bd_sphere):
        mesh = sm.VolumeMesh.from_geometry(
            geom_bd_sphere, sm.GmshVolumeOptions(max_mesh_size=5)
        )
        model = sm.MOABModel.from_mesh(mesh)


class TestMOABVolumeModel:
    @pytest.fixture(scope="class")
    def volume_model(self):
        solid = bd.Solid.make_sphere(10.0)
        geom = sm.Geometry([solid], [""])
        mesh = sm.VolumeMesh.from_geometry(geom, sm.GmshVolumeOptions(max_mesh_size=5))
        return sm.MOABVolumeModel.from_mesh(mesh)

    def test_has_tets(self, volume_model):
        tets = volume_model.tets
        assert isinstance(tets, Range)
        assert not tets.empty()

    def test_no_triangles_in_root(self, volume_model):
        tris = volume_model.triangles
        assert tris.empty()

    def test_write_and_read(self, volume_model, tmp_path):
        out = tmp_path / "volume.h5m"
        volume_model.write(out)
        assert out.exists()
        reloaded = sm.MOABVolumeModel(out)
        assert not reloaded.tets.empty()

    def test_tet_count_reasonable(self, volume_model):
        assert len(volume_model.tets) > 100

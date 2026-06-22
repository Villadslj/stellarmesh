import build123d as bd
import cadquery as cq
import pytest
import stellarmesh as sm


class TestGeometryInitialization:
    @pytest.mark.parametrize(
        "fixture_name",
        [
            "model_bd_layered_torus",
            "model_cq_layered_torus",
            "model_ocp_layered_torus",
        ],
    )
    def test_geometry_init(self, fixture_name, request):
        solids = request.getfixturevalue(fixture_name)
        material_names = ["material"] * len(solids)
        geom = sm.Geometry(solids, material_names)

        if hasattr(solids[0], "wrapped"):
            assert geom.solids == [s.wrapped for s in solids]
        else:
            assert geom.solids == solids
        assert geom.material_names == material_names

    def test_geometry_init_wrong_materials(self, model_bd_layered_torus):
        solids = model_bd_layered_torus
        material_names = ["material"] * (len(solids) - 1)
        with pytest.raises(ValueError):
            sm.Geometry(solids, material_names)


class TestGeometryImportExport:
    def test_step_import_compound(self, model_bd_layered_torus):
        cmp = bd.Compound(model_bd_layered_torus)
        bd.export_step(cmp, "model.step")
        sm.Geometry.from_step("model.step", material_names=[""] * 3)
        with pytest.raises(ValueError):
            sm.Geometry.from_step("model.step", material_names=[""] * 2)

    def test_step_import_solid(self, model_bd_layered_torus):
        bd.export_step(model_bd_layered_torus[0], "layer.step")
        sm.Geometry.from_step("layer.step", material_names=[""])

    def test_step_import_auto_names(self, model_bd_layered_torus):
        cmp = bd.Compound(model_bd_layered_torus)
        bd.export_step(cmp, "model.step")
        geom = sm.Geometry.from_step("model.step")
        assert len(geom.material_names) == 3

    def test_step_import_auto_names_from_named_assembly(self):
        assy = cq.Assembly(name="top")
        assy.add(cq.Workplane().box(1, 1, 1), name="ss")
        assy.add(cq.Workplane().translate((2, 0, 0)).box(1, 1, 1), name="fs")
        assy.save("named.step")

        geom = sm.Geometry.from_step("named.step")

        assert geom.material_names == ["ss", "fs"]

    def test_brep_import_compound(self, model_bd_layered_torus):
        cmp = bd.Compound(model_bd_layered_torus)
        bd.export_brep(cmp, "model.brep")
        sm.Geometry.from_brep("model.brep", material_names=[""] * 3)
        with pytest.raises(ValueError):
            sm.Geometry.from_brep("model.brep", material_names=[""] * 2)

    def test_brep_import_solid(self, model_bd_layered_torus):
        bd.export_brep(model_bd_layered_torus[0], "layer.brep")
        sm.Geometry.from_brep("layer.brep", material_names=[""])


class TestGeometryOperations:
    def test_geometry_imprint(self, geom_bd_layered_torus):
        geom_bd_layered_torus.imprint()

    def test_geometry_imprint_staged(self, geom_bd_layered_torus):
        result = geom_bd_layered_torus.imprint(batch_size=2)
        assert len(result.solids) == len(geom_bd_layered_torus.solids)
        assert result.material_names == geom_bd_layered_torus.material_names

    def test_geometry_imprint_staged_batch_size_equals_solids(
        self, geom_bd_layered_torus
    ):
        result = geom_bd_layered_torus.imprint(batch_size=10)
        assert len(result.solids) == len(geom_bd_layered_torus.solids)

    def test_geometry_imprint_batch_size_validation(self, geom_bd_layered_torus):
        with pytest.raises(ValueError, match="batch_size must be at least 2"):
            geom_bd_layered_torus.imprint(batch_size=1)


class TestMaterialNames:
    def test_get_material_names(self, model_bd_layered_torus):
        material_names = ["mat_a", "mat_b", "mat_c"]
        geom = sm.Geometry(model_bd_layered_torus, material_names=material_names)
        assert geom.get_material_names() == material_names

    def test_get_material_names_returns_copy(self, model_bd_layered_torus):
        material_names = ["mat_a", "mat_b", "mat_c"]
        geom = sm.Geometry(model_bd_layered_torus, material_names=material_names)
        result = geom.get_material_names()
        result[0] = "modified"
        assert geom.get_material_names()[0] == "mat_a"

    def test_set_material_names(self, model_bd_layered_torus):
        material_names = ["mat_a", "mat_b", "mat_c"]
        geom = sm.Geometry(model_bd_layered_torus, material_names=material_names)
        new_names = ["new_a", "new_b", "new_c"]
        geom.set_material_names(new_names)
        assert geom.material_names == new_names

    def test_set_material_names_wrong_length(self, model_bd_layered_torus):
        material_names = ["mat_a", "mat_b", "mat_c"]
        geom = sm.Geometry(model_bd_layered_torus, material_names=material_names)
        with pytest.raises(ValueError, match="does not match"):
            geom.set_material_names(["only_one"])

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
        assert geom.part_names == material_names

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
        assert geom.part_names == ["ss", "fs"]
        assert geom.assembly_names == [["top"], ["top"]]

    def test_step_import_nested_assembly_names(self):
        subassembly = cq.Assembly(name="Cylinder sub")
        subassembly.add(cq.Workplane().box(1, 1, 1), name="ss - Pin")
        subassembly.add(
            cq.Workplane().translate((2, 0, 0)).box(1, 1, 1),
            name="ss - Rod",
        )
        assembly = cq.Assembly(name="Engine")
        assembly.add(subassembly, name="Cylinder sub <1>")
        assembly.save("nested.step")

        geom = sm.Geometry.from_step("nested.step")

        assert geom.material_names == ["ss - Pin", "ss - Rod"]
        assert geom.part_names == ["ss - Pin", "ss - Rod"]
        assert geom.assembly_names == [
            ["Engine", "Cylinder sub"],
            ["Engine", "Cylinder sub"],
        ]

        geom.set_material_names(["ss", "ss"])

        assert geom.part_names == ["ss - Pin", "ss - Rod"]
        selected = geom.select("ss - Pin")
        assert selected.material_names == ["ss"]
        assert selected.part_names == ["ss - Pin"]

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
    @pytest.fixture
    def named_geometry(self):
        solids = [
            bd.Solid.make_box(1, 1, 1),
            bd.Solid.make_box(1, 1, 1).transformed(offset=(2, 0, 0)),
            bd.Solid.make_box(1, 1, 1).transformed(offset=(4, 0, 0)),
        ]
        return sm.Geometry(
            solids,
            ["part_a", "part_b", "part_c"],
            assembly_names=[["assembly"], ["assembly", "sub"], ["other"]],
        )

    def test_select_part(self, named_geometry):
        selected = named_geometry.select("part_b")
        assert selected.material_names == ["part_b"]
        assert selected.part_names == ["part_b"]
        assert selected.assembly_names == [["assembly", "sub"]]

    def test_select_assembly(self, named_geometry):
        selected = named_geometry.select("assembly")
        assert selected.material_names == ["part_a", "part_b"]

    def test_select_multiple_names_deduplicates(self, named_geometry):
        selected = named_geometry.select(["assembly", "part_b", "part_c"])
        assert selected.material_names == ["part_a", "part_b", "part_c"]

    def test_select_unknown_name(self, named_geometry):
        with pytest.raises(KeyError, match="Available names"):
            named_geometry.select("missing")

    def test_select_ambiguous_name(self):
        geometry = sm.Geometry(
            [
                bd.Solid.make_box(1, 1, 1),
                bd.Solid.make_box(1, 1, 1).transformed(offset=(2, 0, 0)),
            ],
            ["shared", "other"],
            assembly_names=[[], ["shared"]],
        )
        with pytest.raises(ValueError, match="both a part and an assembly"):
            geometry.select("shared")

    def test_geometry_imprint(self, geom_bd_layered_torus):
        geom_bd_layered_torus.imprint()

    def test_geometry_imprint_staged(self, geom_bd_layered_torus):
        geom_bd_layered_torus.set_assembly_names(
            [["assembly"]] * len(geom_bd_layered_torus.solids)
        )
        result = geom_bd_layered_torus.imprint(batch_size=2)
        assert len(result.solids) == len(geom_bd_layered_torus.solids)
        assert result.material_names == geom_bd_layered_torus.material_names
        assert result.part_names == geom_bd_layered_torus.part_names
        assert result.assembly_names == geom_bd_layered_torus.assembly_names

    def test_geometry_imprint_staged_batch_size_equals_solids(
        self, geom_bd_layered_torus
    ):
        result = geom_bd_layered_torus.imprint(batch_size=10)
        assert len(result.solids) == len(geom_bd_layered_torus.solids)

    def test_geometry_imprint_batch_size_validation(self, geom_bd_layered_torus):
        with pytest.raises(ValueError, match="batch_size must be at least 2"):
            geom_bd_layered_torus.imprint(batch_size=1)

    def test_sweep_adjacency_matches_pairwise_reference(self):
        bboxes = [
            (0, 0, 0, 2, 2, 2),
            (1, 1, 1, 3, 3, 3),
            (4, 0, 0, 5, 1, 1),
            (5, 1, 1, 6, 2, 2),
            (-3, -3, -3, -2, -2, -2),
        ]
        expected = [set() for _ in bboxes]
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if sm.Geometry._bounding_boxes_overlap(bboxes[i], bboxes[j]):
                    expected[i].add(j)
                    expected[j].add(i)

        assert sm.Geometry()._build_adjacency(len(bboxes), bboxes) == expected


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


class TestPartNames:
    def test_part_names_are_independent_of_materials(self, model_bd_layered_torus):
        geom = sm.Geometry(
            model_bd_layered_torus,
            material_names=["fe", "fe", "ss"],
            part_names=["fe - Block", "fe - Pin", "ss - Tank"],
        )

        geom.set_material_names(["steel", "steel", "steel"])

        assert geom.get_part_names() == [
            "fe - Block",
            "fe - Pin",
            "ss - Tank",
        ]
        assert geom.select("fe - Block").material_names == ["steel"]

    def test_get_part_names_returns_copy(self, model_bd_layered_torus):
        geom = sm.Geometry(
            model_bd_layered_torus,
            material_names=["fe", "fe", "ss"],
            part_names=["fe - Block", "fe - Pin", "ss - Tank"],
        )
        result = geom.get_part_names()
        result[0] = "modified"
        assert geom.part_names[0] == "fe - Block"

    def test_set_part_names_wrong_length(self, model_bd_layered_torus):
        geom = sm.Geometry(
            model_bd_layered_torus,
            material_names=["fe", "fe", "ss"],
        )
        with pytest.raises(ValueError, match="does not match"):
            geom.set_part_names(["only_one"])


class TestAssemblyNames:
    def test_get_assembly_names_returns_deep_copy(self, model_bd_layered_torus):
        geom = sm.Geometry(
            model_bd_layered_torus,
            material_names=["material"] * len(model_bd_layered_torus),
            assembly_names=[["assembly"]] * len(model_bd_layered_torus),
        )
        result = geom.get_assembly_names()
        result[0][0] = "modified"
        assert geom.assembly_names[0] == ["assembly"]

    def test_set_assembly_names_wrong_length(self, model_bd_layered_torus):
        geom = sm.Geometry(
            model_bd_layered_torus,
            material_names=["material"] * len(model_bd_layered_torus),
        )
        with pytest.raises(ValueError, match="does not match"):
            geom.set_assembly_names([["only_one"]])

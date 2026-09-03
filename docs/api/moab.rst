====
MOAB
====
.. module:: stellarmesh

---------------------
MOAB and DAGMC Models
---------------------

Both the :py:class:`MOABModel` and the inherited :py:class:`DAGMCModel` classes can be instantiated from a Stellarmesh :py:class:`Mesh` object using the :py:meth:`from_mesh() <DAGMCModel.from_mesh>` constructor.

Use the :py:meth:`write() <MOABModel.write>` method to write a ``.h5m`` file for import in OpenMC.

See the `Surface Meshing <../notebooks/tutorials/surface_meshing.html#Run-an-OpenMC-Tally>`__ tutorial for a complete example.

Named-part and assembly workflows
---------------------------------

For DAGMC models, volume IDs (``GLOBAL_ID``) can be queried by material/part name and used
directly with OpenMC DAGMC cell filters:

.. code-block:: python

   id_map = dagmc_model.material_to_volume_ids
   cell_ids = id_map["blanket_module"]
   # openmc.CellFilter(cell_ids)

STEP assembly membership is preserved in separate DAGMC ``assembly:`` groups. Query an
assembly to select all volumes contained in every instance of that assembly:

.. code-block:: python

   assembly_map = dagmc_model.assembly_to_volume_ids
   cylinder_cells = assembly_map["Cylinder sub"]

   # Resolve either a part/material or assembly name.
   cylinder_cells = dagmc_model.volume_ids("Cylinder sub")

You can also get tight axis-aligned bounds for a part, assembly, or set of volume IDs:

.. code-block:: python

   part_bounds = dagmc_model.bounding_box("blanket_module")
   assembly_bounds = dagmc_model.bounding_box("Cylinder sub")
   combined_bounds = dagmc_model.bounding_box(cell_ids)


API
---------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    MOABModel
    DAGMCModel
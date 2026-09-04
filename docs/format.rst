======
Format
======

.. module:: stellarmesh
   :no-index:

Stellarmesh consumes Gmsh ``.msh`` v4.1 ASCII files annotated with
physical groups using a URL-style ``key=value`` encoding.

-----------------------
Physical-group encoding
-----------------------

Per-entity metadata is stored as the **name** of a Gmsh physical
group covering exactly one Gmsh entity. The name encodes a URL-style
query string of key/value pairs, parsed by stellarmesh as follows.

Volume groups
=============

::

   tag=<integer>&material=<string>&part=<string>

* ``tag`` — the volume's Gmsh tag (matches the discrete entity tag in
  the same file).
* ``material`` — a slug identifying the volume's material region. DAGMC reads
  ``mat:<slug>`` groups directly through MOAB's fixed 32-byte ``NAME`` tag, so
  the slug is bounded to **28 characters**.
* ``part`` — the persistent CAD part name for the volume. It is independent
  of ``material``, so transport materials can be normalized without losing
  part identity. Part names are not read by DAGMC and may be longer than the
  ``NAME`` tag: the truncated value is written to ``NAME`` and the full name to
  an auxiliary ``STELLARMESH_NAME`` tag that stellarmesh reads back.

Every volume entity in the file must carry exactly one such physical
group.

Assembly membership is stored in additional multi-entity physical groups
named ``assembly:<name>``.

Surface groups
==============

::

   tag=<integer>&forward_volume=<integer>&reverse_volume=<integer>

* ``tag`` — the surface's Gmsh tag.
* ``forward_volume`` — the volume tag on the surface's outward-normal
  side.
* ``reverse_volume`` — the volume tag on the other side. Use ``0`` for
  exterior surfaces (boundary of the model with vacuum).

Edges and vertices
==================

Discrete entities only; **no physical groups**. Stellarmesh ignores
edge and vertex annotations if present.

------------------
Consumer behaviour
------------------

When stellarmesh reads a conforming ``.msh`` file:

* Each volume's ``material`` slug becomes a DAGMC ``mat:<slug>`` group
  in the output ``.h5m`` file.
* Each volume's ``part`` value becomes a DAGMC ``part:<name>`` group.
* ``assembly:<name>`` groups are preserved for named assembly selection.
* ``part:`` and ``assembly:`` group names longer than 31 bytes are truncated in
  the MOAB ``NAME`` tag (with a unique ``~<n>`` suffix on collision) and stored
  in full in ``STELLARMESH_NAME``. ``mat:`` groups are never truncated and raise
  a ``ValueError`` instead, because DAGMC must read them verbatim.
* Surface ``forward_volume`` / ``reverse_volume`` populate DAGMC's
  surface-sense relationships.
* Downstream tooling (e.g. OpenMC) maps the ``mat:<slug>`` groups to
  ``openmc.Material`` instances. Stellarmesh does not perform that
  mapping itself.

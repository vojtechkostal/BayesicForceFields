# Analyze Configuration

`bff analyze` joins sampled and reference systems by stable `system_id`, never
by list position. Both sections must contain the same unique ID set.

```yaml
training_samples:
  manifest: ../03-sample/samples.yaml
  systems:
    - system_id: acetate
    - system_id: acetate-contact
  frames: {start: 1, stop: null, step: 1}
  workers: -1
  progress_stride: 10

reference:
  systems:
    - system_id: acetate
      inputs:
        topology: ../reference/acetate/topology.top
        coordinates: ../reference/acetate/coordinates.gro
        trajectory: ../reference/acetate/trajectory.xtc
    - system_id: acetate-contact
      inputs:
        topology: ../reference/contact/topology.top
        coordinates: ../reference/contact/coordinates.gro
        trajectory: ../reference/contact/trajectory.xtc
        pmf: ../reference/contact/profile.pmf
  frames: {start: 1, stop: null, step: 1}

routines:
  - name: acetate-water-rdf
    type: rdf
    systems: [acetate, acetate-contact]
    selections:
      group_a: "resname ACE and name O1 O2"
      group_b: "resname SOL and name O*"
    options:
      range: [1.0, 7.0]
      bins: 200
      pbc: true
      update_selections: false
      smooth: false
  - name: contact-pmf
    callable: ../inputs/pmf.py:load_profile
    loader: files
    systems: [acetate-contact]
    inputs: [pmf]

run:
  in_memory: true
  gc_collect: false
  maxtasksperchild: 100
output:
  directory: ./qoi
  write_raw: false
```

Built-ins are `rdf` and `hydrogen_bonds`. RDF requires `group_a` and
`group_b`; hydrogen bonds require `donors`, `hydrogens`, and `acceptors`.
Selections are full MDAnalysis expressions. Static selections are the default;
`update_selections: true` reevaluates them each frame. Empty selections,
missing bonds, and invalid PBC boxes are errors.

A custom `loader: files` callable receives keyword arguments `inputs`,
`system_id`, `sample_id`, and `options`. A custom `loader: mdanalysis` callable
receives `universe`, `frames`, both IDs, and `options`. Every routine returns
exactly one `QoI`; the configured routine name is authoritative.

Outputs are `qoi/<routine-name>.pt`, `analyze.log`, and optional
`qoi/raw.json`. Dataset metadata records `system_ids`.

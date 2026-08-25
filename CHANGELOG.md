## `0.4.1` - 2026-08-25

### Changed

- Simplified RDF calculation so selection handling and atom-type expansion
  occur in the QoI adapter while the numerical kernel operates directly on
  MDAnalysis AtomGroups. Dynamic selections and numerical behavior are
  preserved.
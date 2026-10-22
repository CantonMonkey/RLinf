# CantonMonkey navigation development

This branch adds the navigation extension under
[`extensions/navigation`](extensions/navigation/README.md).

- Upstream runtime baseline: `0505431899574619da86f551bad70b71e0ea2177`.
- Imported local source: `navrlt-lightnav-v1` at
  `b9c2af0ef32f41f6065896ee5d6dbf4e494ce169`.
- Working branch: `navigation-vln`; the fork's `main` follows upstream.
- The extension includes navigation contracts, Habitat services, RLinf staging
  modules, StreamVLN/LightNav/Navigation-RLT code, launch templates and tests.
- Source provenance and file digests: [EXPORT_PROVENANCE.json](extensions/navigation/EXPORT_PROVENANCE.json).

Start with the extension README. Deployment templates use example users, paths,
network addresses, GPU identities and storage sources. Historical run manifests
must be regenerated for the deployed source. Datasets, pretrained weights, run
outputs and private machine configuration are provisioned separately.

The export retains upstream ancestry while omitting the independent planning
repository's private history. Existing upstream modules are unchanged: selected
integration files must be installed deliberately for the intended execution path.

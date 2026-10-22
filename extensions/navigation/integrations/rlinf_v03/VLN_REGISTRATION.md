# RLinf v0.3 VLN policy registration

This patch targets the isolated A800 `RLinf-v0.3` checkout at commit
`0505431899574619da86f551bad70b71e0ea2177`. The three staged files were
copied byte-for-byte from that checkout on 2026-09-24, then changed only to add
the `vln_policy` registration, rollout argument branches and RGB batch inference.
They were installed in the A800 private checkout on 2026-09-24; see
`VLN_A800_INSTALL.md`. Do not apply these complete-file copies to a checkout
whose corresponding source files have changed without reviewing the diff.

| Local staged file | Exact destination under `repos/RLinf-v0.3/` | Source SHA256 before patch |
| --- | --- | --- |
| `config_vln.py` | `rlinf/config.py` | `bef76609312e3d81296dee3de39685df512592c5511002b729357ab1a7006662` |
| `models_init_vln.py` | `rlinf/models/__init__.py` | `b69209448b426d66d671103ee739f7453dc332a82c81fede3d711c26de4986ee` |
| `huggingface_worker_vln.py` | `rlinf/workers/rollout/hf/huggingface_worker.py` | `480d3797cb86b55cfecee04c5020f9c5ef9bea60f7b5e777d494e277d9ddd270` |

The separate policy implementation `vln_policy.py` is intended for
`rlinf/models/embodiment/vln_policy.py`, where the new lazy builder imports
`rlinf.models.embodiment.vln_policy.get_model`. Its implementation is owned by
the separate VLN policy workstream and was not edited here.

`config_vln.py` registers `SupportedModel.VLN_POLICY` and adds it to
`EMBODIED_MODEL`. `models_init_vln.py` adds one lazy builder to the existing
model registry. `huggingface_worker_vln.py` adds the new type to the existing
`mode` and `return_obs` branches, so rollout supplies the parameters expected
by the current policy, and recognizes RGB when inferring the environment batch
size. The existing model branches remain present. `embodied_io_struct_vln.py`
also preserves exactly the three VLN observation tensors through RLinf's
`EnvOutput.to_dict()` boundary; without this additional patch rollout would
receive only legacy empty observation keys.

Local, dependency-free check:

```powershell
cd C:\path\to\navigation-project\rlinf-nav
python -B -m unittest discover -s tests -p 'test_vln_registry_staged.py' -v
```

The test executes the staged model registry with a stub VLN builder and checks
the supported/embodied type and rollout branch wiring. It does not initialize
RLinf, Ray, CUDA, Habitat, or FSDP. In the inspected generic actor PPO path,
`_process_micro_batch` forwards `forward_inputs` to the selected model; no
additional certain model-type branch was found. Hydra composition and the real
Habitat environment bridge now pass on A800; the installed model/rollout CPU
handshake and Ray/FSDP training still require runtime checks.

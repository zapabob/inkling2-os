# Known limitations

The Windows adapter runs the native process per admitted request. It uses the model's embedded template and Hugging Face tokenizer, and sends numeric token IDs to the CUDA process. Native generation currently supports plain-text messages with reasoning disabled and a 4,096-token engine context. It buffers the answer before emitting OpenAI-compatible events. Model-level accuracy and long-context qualification remain separate from tokenizer conformance.

The Linux engine builds for `sm_120a`. Physical GPU-backed boot, model generation, scheduler profiles, comparative performance and long context require hardware qualification. The live ISO’s companion state is volatile. Dynamic LoRA remains an adapter-state experiment, without a confirmed model-weight update path.

The built-in Browser renders public IPv4 HTTPS documents and follows up to three validated redirects. Sites that require JavaScript, authentication, video or other active resources are outside the document reader. Finder offers read-only previews of approved text files. The Windows console runs within Windows' existing network and filesystem policy. The private Linux disk requires UEFI and provides a graphical view of the local console and persistent storage. Wine runs command-line applications through a private virtual display.

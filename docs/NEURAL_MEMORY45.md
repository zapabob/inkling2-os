# Neural Memory45

Neural Memory45 combines explicit records, a deterministic product-quantised index and two banks of quantised associative weights. The snapshot container has a maximum size of 47,185,920 bytes (45 MiB). Admission limits apply before encoding or updating weights. These limits describe stored records and snapshot size. Process memory also includes Rust allocations, the FP32 codebook, temporary candidate copies and serialisation buffers.

## Admission and storage budgets

A record contains up to 4,096 UTF-8 bytes of text and 256 bytes of provenance, with finite importance in the range 0 to 1. At most 4,096 records are admitted. Admission also enforces the 12 MiB exact-record budget using conservative per-record overhead. Exhausted budgets and invalid input return an error without assigning an ID or changing weights.

| Region | Budget in MiB | Current use |
|---|---:|---|
| Exact records | 12 | Text, provenance, IDs, timestamps and record accounting |
| Product quantisation | 6 | 512 dimensions, 32 subspaces and 256 centroids per subspace, with a 32-byte code per record |
| Associative weights | 10 | Two banks of 32 shards, each with 512 by 256 INT8 weights and FP16 row scales |
| Dynamic LoRA | 8 | Reserved independently of the record snapshot |
| Journal and replay | 5 | Reserved |
| Metadata | 2 | Two checksummed copies of the snapshot header |
| Emergency space | 2 | Reserved |
| Total | 45 | Compile-time sum |

The running codebook uses FP32 values. Its equivalent FP16 footprint is 262,144 bytes. The 45 MiB region allocation reserves capacity for future components, and the current snapshot serialises explicit records, PQ codes, codebook seed and version. Associative weights are reconstructed from eligible records when loading.

## Associative updates

The encoder produces normalised 256-dimensional keys and 512-dimensional values. An update selects the two highest-scoring shards, predicts the value from each key, and applies a bounded rank-one correction. Finite learning-rate and gate values must lie between 0 and 1, and decay must lie between 0 and 0.25. Invalid numerical inputs leave the shard unchanged. Binary16 conversion preserves subnormal row scales with round-to-nearest-even, so small first updates remain observable.

A candidate bank is promoted when its reconstruction error on the current record does not increase. This check measures that record alone. Retention across other tasks requires a separate measured evaluation. Untrusted observations, unverified text, denied items and sensitive records marked non-deletable cannot fold into the fast weights.

## Recall and deletion

Recall combines the query with the associative prediction and scores explicit records. Results include the stored source ID, text, provenance, timestamp and verification status. Confidence is a retrieval score, and the stored verification label remains provenance metadata.

Deletion removes the explicit record and its PQ code, then rebuilds both weight banks from the remaining eligible records. Persisting the resulting snapshot makes deletion survive a restart. The shell uses save-before-commit transactions for API mutations.

## Snapshot integrity and recovery

Saving writes an exclusively created temporary file in the destination directory, flushes its contents and atomically renames it over the snapshot. Failed writes leave the current snapshot intact and attempt temporary-file cleanup. Two header copies refer to the same payload, so one damaged header can be recovered using the other. They contain no previous payload generation.

Loading bounds the file read to 45 MiB, validates the payload checksum, record count, region use, UTF-8 lengths, unique ordered IDs, finite importance and learning flags before rebuilding weights. Unexpected trailing bytes are rejected. The FNV-1a checksums detect accidental corruption. Snapshot files require the same filesystem access controls as other companion state.

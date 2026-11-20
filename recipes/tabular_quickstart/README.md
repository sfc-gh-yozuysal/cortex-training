# Tabular foundation models

A tabular foundation model predicts the targets of query rows from labeled
context rows of the same table in one forward pass. On Cortex Training it runs
as a `tabular` training job: the server runs only the network, the model
library's preprocessing runs in your process, and each request carries one
entry per network call
([REST API section 8.7](../../docs/reference/rest-api.md#87-tabular-foundation-models),
[section 10.6](../../docs/reference/rest-api.md#106-tabular-models)).

| Recipe | Model | Loss |
|---|---|---|
| [`tabpfn_finetune`](tabpfn_finetune/README.md) | TabPFN v2 classifier (`tabpfn_v2`) | `weighted_cross_entropy`, or a client loss through `output_gradient` |
| [`kumo_finetune`](kumo_finetune/README.md) | NVIDIA Kumo Tabular (`kumo_tabular`) | `weighted_cross_entropy` (classification), pinball through `output_gradient` (regression) |

Both use `cortex_training.client.tabular` to build frames:

```python
from cortex_training.client import tabular

call = tabular.Call(x=x, y=y, categorical_inds=[[]])            # x [R, T, F], y [R_context, T]
payload = tabular.build_forward_backward_payload([call], [targets], num_classes=10)
client.poll_request(job_id, client.forward_backward(job_id, payload))
```

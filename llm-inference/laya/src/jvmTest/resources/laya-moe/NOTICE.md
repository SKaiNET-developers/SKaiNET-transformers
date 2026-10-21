# layaMOE evaluation fixtures

`eval/*.json` (108 hand-labelled cases in nine domains) and `moe_eval.json` (the published answers of
the original head, the oracle routing and the routed mixture on those cases, PyTorch FP32 on CPU) are
copied unchanged from the layaMOE repository at commit `b5e8c932a99204590c32f2cf98847ed05a182555`:
https://github.com/vishalmysore/layaMOE/tree/b5e8c932a99204590c32f2cf98847ed05a182555 (`data/eval/`,
`results/moe_eval.json`).

Copyright 2026 vishalmysore. Licensed under the Apache License, Version 2.0; the license text is in
`LICENSE-APACHE-2.0.txt` next to this file. layaMOE is an unofficial experiment on top of the
`laya-typed-decisions` checkpoint by ConvAI Innovations (Apache-2.0) and is not affiliated with or
endorsed by ConvAI Innovations, Answer.AI or LightOn. The fixtures are test data only and are not part
of the published artifact; the expert weights they were produced with are downloaded by the gated tests
from https://huggingface.co/VishalMysore/layaMOE and are not redistributed here.

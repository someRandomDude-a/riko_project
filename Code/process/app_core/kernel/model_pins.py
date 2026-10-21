"""Pinned remote models whose repositories ship Python code that runs inside Riko (emotion/teacher.py loads it).

Julia-1 (SupersonicLabs/Julia-1) is downloaded with huggingface_hub and its `julia` package is imported from the snapshot,
so a branch would let an upstream push (or a compromised account) change the code Riko runs. The download is pinned to a
commit, and the snapshot's `julia/**/*.py` must match JULIA_CODE_SHA256 before anything imports it.

Recorded 2026-10-16 from https://huggingface.co/api/models/SupersonicLabs/Julia-1 (`sha`, the head of main then, its
newest commit: 2026-09-26T22:03:07Z "Preserve Boolean criteria and add pinned accuracy reproduction") and
/api/models/SupersonicLabs/Julia-1/revision/<sha>?blobs=true: each file below was fetched from /resolve/<sha>/<file>, its
git blob hash matched the API's blobId, and its SHA-256 is recorded here. model.safetensors at the pin: LFS sha256
df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72, 577,189,056 bytes. To move the pin, record the new
commit and its files the same way (a changed file fails the check until it is listed).
"""
import re

JULIA_MODEL_ID = 'SupersonicLabs/Julia-1'
JULIA_REVISION = 'a85b127321d580d65176c89ced8273f305745d85'
JULIA_CODE_SHA256 = {
    'julia/__init__.py': '02485eb4dd7dadd13907380d1a9c3398709bd8af54b3e0e12b3061a3ebb5c1da',
    'julia/cuda.py': '138ab63182f25473ce3886296c47ac44e94fc50ae5270927ae4c373f7bb49137',
    'julia/data.py': 'e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e',
    'julia/inference.py': '79b4e716365a6e07da4580ead725d5b76d06ef3a824a3ec23738feba64704b36',
    'julia/model.py': 'ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81',
    'julia/probabilities.py': '2a0197d2e0fa5a3c4b06b93599a85706724df7b0d2cc821a13ed293aef59f206',
    'julia/typed.py': 'ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d',
    'julia/router/__init__.py': '832cf0a44569941efba66dfa7099b645fdb1084082e40c6115bb0ac5edd78a96',
    'julia/router/build.py': 'ea059c3a97425cadb1068fe41626b6d3bcd77e1efccfc154dbde29352c432ed9',
    'julia/router/encoder.py': 'df25efee2ed916d52af9c4e9c0d98e80854ca70bbfc8140f1de871f290fd3238',
    'julia/router/engine.py': '91bb30987ff8626ed1610955c6fd796939a5d271e2405b73f8674ccb947eb931',
    'julia/router/native.py': '42bdba5ea7d874e1f53b2f17e99e731789611b5bd6734f687eaa2b0262c5d7ae',
    'julia/router/router.py': '624b0b82fd34e4c6874a37595fa91a1d79553d0ca9d8111ef8330ed805f6be7c',
    'julia/router/specialize.py': '4ba766a45785da0f48898143eb04e13bdf736019a8ba76b71ad84a4bfec8b2d9',
    'julia/router/transformer.py': '0ce04fab361ca4e8bed636c615e5a6e178fa1a52332db34457e5235c667e1786',
    'julia/router/tests/synthetic.py': 'f2b37bd6ceaf184449f1037b6aa8fda3bcb6263c76b3d11453c11c18d23d0196',
    'julia/router/tests/test_engine.py': 'eb91ee1963400cb732350dadb7f5e39cb68917c6314ca8039e24974310364d52',
    'julia/router/tests/test_router.py': 'e4df6d1f7d0962d0b5b119572d2d3897f1f4cadaa72600f26f3e72090a49172b',
}
COMMIT = re.compile(r'[0-9a-f]{40}')


def commit(path, value):
    """value when it is a full commit hash; a branch, tag or short hash raises ValueError naming the setting at path."""
    if isinstance(value, str) and COMMIT.fullmatch(value): return value
    raise ValueError(f'{path} must be a full 40-character commit hash, not {value!r}: a branch or tag can change the code that runs')

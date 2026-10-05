# Upstream relationship / 与作者代码的关系

`Ther-nullptr/piR2` is an independent experiment repository, not the paper authors' official repository and not a GitHub fork. The runtime directory `upstream/` is a clone of [the official parent repository](https://github.com/pi-r2-flow/pi-r2-flow), whose `learning/Isaac-GR00T` submodule supplies the actual πR²/Flow implementation.

The parent is pinned to `3af52ca400a6ec7d141416879aa531a6f62f697a`, and the GR00T fork to `2a39d591a3af24c42bb1d1c9cd708a2dddac0600`. Our repository owns the experiment entrypoints, LIBERO adaptation, timing/evaluation tooling and explicitly recorded corrections. It does not present the algorithm as newly invented here.

Source locations and commits live in [sources.lock.json](../sources.lock.json). Dependencies are downloaded to ignored directories rather than importing nested `.git` histories or copying complete source trees into commits. Local corrections are tracked as patches with the relevant upstream license and notice; the preparation command checks the exact base and fails on incompatible changes.

The official release focuses on xArm6 + XHand deployment and GR00T training. Our LIBERO experiments and reconstructed Leap data therefore require separate validation; they are not the authors' original experiment assets.

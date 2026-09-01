# TraceGuard curated dataset — audit report

Total curated images: **86498** (after 114 sha256 dedupe drops, 0 Pillow verification drops)

## Counts by category x label x split

| category | label | split | count |
|---|---|---|---|
| scene | 1 | train | 19521 |
| face | 1 | train | 13471 |
| scene | 1 | eval_genimagepp | 12000 |
| scene | 0 | train | 11050 |
| face | 0 | train | 10188 |
| scene | 1 | eval_tampered | 4992 |
| scene | 1 | val | 3479 |
| face | 1 | eval_face_unseen | 3006 |
| face | 0 | eval_face_cdf | 2664 |
| face | 1 | val | 2377 |
| scene | 0 | val | 1950 |
| face | 0 | val | 1800 |

## Counts by source x label x split

| source | label | split | count |
|---|---|---|---|
| df40 | 1 | train | 13471 |
| genimagepp | 1 | eval_genimagepp | 12000 |
| df40 | 0 | train | 10188 |
| wildfake | 1 | train | 8500 |
| commfor_small | 0 | train | 6800 |
| commfor_small | 1 | train | 6771 |
| sid_set | 1 | eval_tampered | 4992 |
| sid_set | 0 | train | 4250 |
| sid_set | 1 | train | 4250 |
| df40 | 1 | eval_face_unseen | 3006 |
| df40 | 0 | eval_face_cdf | 2664 |
| df40 | 1 | val | 2377 |
| df40 | 0 | val | 1800 |
| wildfake | 1 | val | 1500 |
| commfor_small | 1 | val | 1229 |
| commfor_small | 0 | val | 1200 |
| sid_set | 0 | val | 750 |
| sid_set | 1 | val | 750 |

## Format histogram by source x label

| source | label | format | count |
|---|---|---|---|
| df40 | 1 | png | 18854 |
| df40 | 0 | png | 14652 |
| genimagepp | 1 | png | 12000 |
| wildfake | 1 | png | 9974 |
| commfor_small | 0 | jpeg | 8000 |
| commfor_small | 1 | png | 8000 |
| sid_set | 1 | png | 5744 |
| sid_set | 0 | jpeg | 4999 |
| sid_set | 1 | jpeg | 4248 |
| wildfake | 1 | jpeg | 26 |
| sid_set | 0 | png | 1 |

## Resolution histogram by source (top 8 each)

- **commfor_small**: 512x512: 8000, 640x480: 1720, 640x427: 1053, 480x640: 574, 500x375: 360, 640x426: 333, 427x640: 266, 640x428: 220, other: 3474
- **df40**: 256x256: 29089, 1024x1024: 1487, 512x512: 1484, 259x259: 1446
- **genimagepp**: 1024x1024: 12000
- **sid_set**: 1024x1024: 10050, 1024x768: 1211, 1024x683: 860, 768x1024: 262, 683x1024: 199, 1024x680: 192, 1024x681: 183, 1024x682: 142, other: 1893
- **wildfake**: 512x512: 2557, 1024x1024: 1480, 256x256: 947, 896x512: 670, 1792x1024: 444, 512x768: 306, 448x256: 305, 512x896: 296, other: 2995

## Bias warnings

- ⚠️ commfor_small label=0: 8000/8000 (100%) are 'jpeg' — a detector could key on format instead of content.
- ⚠️ commfor_small label=1: 8000/8000 (100%) are 'png' — a detector could key on format instead of content.
- ⚠️ df40 label=0: 14652/14652 (100%) are 'png' — a detector could key on format instead of content.
- ⚠️ df40 label=1: 18854/18854 (100%) are 'png' — a detector could key on format instead of content.
- ⚠️ genimagepp label=1: 12000/12000 (100%) are 'png' — a detector could key on format instead of content.
- ⚠️ sid_set label=0: 4999/5000 (100%) are 'jpeg' — a detector could key on format instead of content.
- ⚠️ wildfake label=1: 9974/10000 (100%) are 'png' — a detector could key on format instead of content.

## Verification drops

- none

## sha256 duplicate drops

- kept images/df40/6be83920653831a8.png, dropped duplicate images/df40/20da07099467064d.png (simswap/frames/001_870/459.png)
- kept images/df40/06b50777010be0b4.png, dropped duplicate images/df40/2f6d64cbc7b0a677.png (simswap/frames/226_491/137.png)
- kept images/df40/4c92ed051b1fd883.png, dropped duplicate images/df40/87a9585d84d484a1.png (simswap/frames/444_655/281.png)
- kept images/df40/e97312c57decbe02.png, dropped duplicate images/df40/a8c85a2819362888.png (simswap/frames/526_436/513.png)
- kept images/df40/450a85ef5dc36867.png, dropped duplicate images/df40/be410915e744d762.png (simswap/frames/639_841/426.png)
- kept images/df40/c9707d381c5ce3bc.png, dropped duplicate images/df40/8add940b7a79eb6e.png (simswap/frames/690_689/159.png)
- kept images/df40/a8181e07a94c2a64.png, dropped duplicate images/df40/0963edbbef8a0a5a.png (simswap/frames/710_788/182.png)
- kept images/df40/cda1f766a6acd938.png, dropped duplicate images/df40/060d72a459b3284f.png (simswap/frames/764_850/083.png)
- kept images/df40/2cc900af109a7b41.png, dropped duplicate images/df40/a9271d598c4b61ad.png (simswap/frames/799_809/316.png)
- kept images/df40/d2958cad456faa48.png, dropped duplicate images/df40/db2e062764a0f308.png (DiT/243/6060.png)
- kept images/df40/4244fa86195a37f6.png, dropped duplicate images/df40/8b0a8898c85c78c6.png (DiT/393/3530.png)
- kept images/df40/3460c314de15372d.png, dropped duplicate images/df40/f14fcda941abf9c8.png (DiT/500/19103.png)
- kept images/df40/2a7d841b852c7fa7.png, dropped duplicate images/df40/f3f589607917a487.png (DiT/520/3555.png)
- kept images/df40/ecbe4cd9d7699854.png, dropped duplicate images/df40/ab19a7852af37981.png (DiT/530/5597.png)
- kept images/df40/eaf5362840884368.png, dropped duplicate images/df40/b3ece1f9c48ae06c.png (DiT/799/12708.png)
- kept images/df40/8edb35718c626516.png, dropped duplicate images/df40/2f71ed485ed4cdad.png (DiT/831/16223.png)
- kept images/df40/9012cafe79b9424b.png, dropped duplicate images/df40/a81b73c2692bc1c1.png (DiT/840/4240.png)
- kept images/df40/a3b17d29f1e98476.png, dropped duplicate images/df40/a2bef1cb308dae38.png (DiT/850/6820.png)
- kept images/df40/b9a839ddf7a57905.png, dropped duplicate images/df40/0586dc1d55033ce2.png (DiT/870/3633.png)
- kept images/df40/1a37f9336da9afaa.png, dropped duplicate images/df40/127ff7b1a02c5ac4.png (DiT/873/9631.png)
- kept images/df40/f0f8367dbe9e54b6.png, dropped duplicate images/df40/e4a42ed1ee35159d.png (DiT/883/5017.png)
- kept images/df40/7bdc9d7afe453a9e.png, dropped duplicate images/df40/4eb367126dc6beb6.png (DiT/957/20523.png)
- kept images/df40/e6fa9c2e2135a017.png, dropped duplicate images/df40/969b5883102875a8.png (DiT/966/12491.png)
- kept images/df40/dace1f4c995af147.png, dropped duplicate images/df40/9d9ca23f662a3000.png (pixart/276/472_277.png)
- kept images/df40/b875e5a224897f6e.png, dropped duplicate images/df40/146e0e62a56ffbf9.png (pixart/303/734_743.png)
- kept images/df40/a9b5a2dba36da1d0.png, dropped duplicate images/df40/29e03e247d11abcc.png (pixart/506/085_201.png)
- kept images/df40/bb1fe63ed52db792.png, dropped duplicate images/df40/f4551f52b64d042c.png (pixart/530/736_106.png)
- kept images/df40/7ba6a290366074f1.png, dropped duplicate images/df40/b17e8385460c4549.png (pixart/613/914_019.png)
- kept images/df40/364696b04f73fa80.png, dropped duplicate images/df40/814969924b88fac7.png (pixart/645/885_328.png)
- kept images/df40/de999558d0929e28.png, dropped duplicate images/df40/9e0b386ea1a6a64e.png (pixart/761/831_210.png)
- kept images/df40/265a6c2908bc2e10.png, dropped duplicate images/df40/0eb8b76ddab0e866.png (pixart/763/118_079.png)
- kept images/df40/cd5715ffb3ccd219.png, dropped duplicate images/df40/22628ec2dfe4c507.png (pixart/819/748_040.png)
- kept images/df40/c600f19b42e1c42f.png, dropped duplicate images/df40/14f69fd39ee0ed9d.png (pixart/828/867_432.png)
- kept images/df40/d48f1b89c16ab13d.png, dropped duplicate images/df40/debf0854d67b6239.png (pixart/889/406_263.png)
- kept images/df40/a22de031f9129279.png, dropped duplicate images/df40/3e5580e1be08fb6d.png (pixart/894/893_121.png)
- kept images/df40/55af5e221106c1b1.png, dropped duplicate images/df40/e749d79eb807074f.png (pixart/977/792_258.png)
- kept images/df40/179da364a2e4b3fd.png, dropped duplicate images/df40/e1e9041c58e3ed93.png (RDDM/207/sample-25.png)
- kept images/df40/0cf02438e6fb2131.png, dropped duplicate images/df40/3ca7ffe4943bc74a.png (RDDM/243/sample-100.png)
- kept images/df40/0820aca8e8b0c54e.png, dropped duplicate images/df40/4c075e29ba9c9c35.png (RDDM/246/sample-7517.png)
- kept images/df40/ef817ae3c31c572b.png, dropped duplicate images/df40/92ca369236089f52.png (RDDM/260/sample-8087.png)
- kept images/df40/58c4ec99bfd4f80f.png, dropped duplicate images/df40/9ee24575adb22701.png (RDDM/285/sample-14506.png)
- kept images/df40/f8fac13eb2fa85fb.png, dropped duplicate images/df40/d7935615460d9ffb.png (RDDM/383/sample-4547.png)
- kept images/df40/9c156f09fd7d2c1c.png, dropped duplicate images/df40/01b7d4bb2408272c.png (RDDM/395/sample-5846.png)
- kept images/df40/52962dc62bb923a0.png, dropped duplicate images/df40/21f185999e7ffb50.png (RDDM/437/sample-2141.png)
- kept images/df40/388b80e315d078b8.png, dropped duplicate images/df40/ae787a6aa5cc0bbc.png (RDDM/437/sample-6248.png)
- kept images/df40/6e7f4ec9136ea1e3.png, dropped duplicate images/df40/ec6309663de6f969.png (RDDM/475/sample-3076.png)
- kept images/df40/210463e676ad784a.png, dropped duplicate images/df40/738b443076fb5af5.png (RDDM/487/sample-8370.png)
- kept images/df40/c925d285343bc268.png, dropped duplicate images/df40/b9f1e2ecfeffa409.png (RDDM/498/sample-1615.png)
- kept images/df40/0d04e9103a504c7b.png, dropped duplicate images/df40/909b53429862e170.png (RDDM/500/sample-2034.png)
- kept images/df40/d2b6706a111e36d6.png, dropped duplicate images/df40/24266b0b136a22a5.png (RDDM/520/sample-17990.png)
- … and 64 more

## Deviations from the original sampling plan (user-approved)

- e4e dropped from train methods: e4e.zip absent from the DF40_train download (JSON only). 11 train methods remain.
- eval_face_cdf contains cdf reals only: DF40's cdf-domain fake frames belong to the separate test release, which is not on disk.
- eval_face_unseen widened with danet, fomm, hyperreenact (~500 each) to compensate for the missing cdf fakes.
- sd2.1, simswap, uniface (train) and SiT (eval) sampled from zip namelists (their dataset_json files are absent); train methods restricted to FF++ train video ids via first video-id token.
- ffpp reals sampled at 12 frames/video (999 videos on disk; 3/video could only reach ~3k of the ~12k target).
- cdf reals: 888 videos x 3 frames/video = 2664 (< nominal 3000 target).
- commfor reals split 85/15 at random (seeded): real rows carry real_source='N/A', so the planned model->real_source linkage does not exist in the data.

## Addendum (2026-08-31, post-build verification)

- **commfor_small fake val split verified model_name-disjoint**: 177 model_names in train, 31 in val, overlap = 0 (checked via meta.csv join AND the manifest generator_or_method column; the two agree on all 8000 fake rows). No re-cut was needed; manifest.csv unchanged. The paired commfor reals remain randomly split 85/15 (seeded) because their real_source is uniformly N/A - they cannot be tied to held-out models.
- **Face-editing (FE) family deferral**: e4e was the only face-editing method in the train plan and its archive is absent from the DF40_train download, so the curated train set contains NO face-editing family. Detectors trained on this set have an untested blind spot for FE-style manipulations; revisit when the e4e (or other FE) archives are obtainable.
- **eval_face_cdf semantics**: this slice is REALS-ONLY (2664 Celeb-DF-v2 frames). It measures the false-positive rate under real-domain shift; it cannot produce AUC/recall on fakes. Do not average it with the other eval slices as if it were a balanced benchmark.
- **Manifest spot-check**: 5 random rows per category x split (seed 42) verified on disk with matching sha256 - see pipeline log for the row list.
- **Contact sheets**: audit/contact_<source>.png, one 10x10 random grid per source (seed 42).

## Addendum: realism filter (2026-08-31)

- **Mechanism**: open_clip ViT-B-32 (laion2b_s34b_b79k) zero-shot,
  realism_score = softmax(100*[sim(photo prompts), sim(stylized prompts)])[photo].
  Photo prompts: ['a photograph', 'a photo taken with a camera', 'a realistic photograph of a scene']. Stylized prompts: ['an anime illustration', 'a cartoon drawing', 'a digital painting', 'concept art', 'a 3d render', 'an illustration'].
- **Threshold**: 0.5. Applied to commfor_small fakes and wildfake only.
- **Policy**: tag, don't delete. New manifest columns style
  (realistic/stylized, blank = not scored) and realism_score. Stylized rows
  formerly in train/val moved to split=eval_stylized (6599
  wildfake + 2521 commfor rows). Files remain on disk.
- **Backfill**: 9120 replacement fakes ingested from unsampled
  remainders (wildfake part_1.zip; CommunityForensics-Small parquet files
  ['11', '12', '13', '14'], provenance recorded as file/rowgroup/row in
  orig_relpath) to restore the original train/val targets. commfor backfill
  keeps the model_name-disjoint val (new models: seeded 15% draw).
  Remaining deficits if any: {}.
- **Sanity sheet**: audit/realism_borderline.png (kept-just-above vs
  tagged-just-below the threshold).
- commfor reals were NOT filtered (they are camera photos by construction);
  sid_set/genimagepp/df40 untouched per scope decision.

## Addendum: real-diversity patch (post-smoke-test, 2026-09-01)

- predict_v2 smoke test exposed a scene-head blind spot: 5/5 pristine held-out
  real photos (CommunityForensics file 160 reals, never in the curated set)
  scored 0.91-0.999 fake. Diagnosis: curated scene reals skew compressed
  web-quality vs clean sharp fakes -> "pristine = fake" shortcut invisible to
  val (same real sources).
- Patch: +6000 pristine reals (931 val) ingested from unused
  CommunityForensics files 160/170/180 as scene train/val label-0 rows
  (source_key commfor_realpatch, provenance file/rowgroup/row). Scene head
  retrained; pre-patch head kept as results/scene_head_prepatch.pkl for
  comparison. Additive only - no rows removed or relabeled.

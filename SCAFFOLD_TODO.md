# M-100 scaffold TODO

Scaffolded from template: `M-043_ddr_lesion_bbox_detection_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=vindrcxr_chest_abnormality_detection, s3_prefix=M-100_VinDr-CXR/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-043_ddr_lesion_bbox_detection_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This chest X-ray. Detect and draw bounding boxes around thoracic abnormalities from VinDr-CXR's 22 disease classes (Aortic enlargement, Cardiomegaly, Pleural effusion, etc.).

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.

# Natural Stories XGLM-small reproduction

This branch first reproduces the released XGLM-small, shift-3 Natural Stories
encoding result before adding any AuriStream analysis. The upstream scripts and
released result files remain unchanged.

The focused runner is `reproduce_xglm_small.py`. It preserves the relevant
operations from the upstream analysis:

- `facebook/xglm-564M` word representations, including the embedding state;
- averaging subword states into one vector per transcript word;
- averaging words into the released two-second response bins;
- train-fold-only standardization of model features and fMRI responses;
- RidgeCV with the original alpha grid;
- leave-one-story-out prediction and held-out-story Pearson correlation; and
- direct comparison of every layer-by-story value with the released
  `results/cross_story_3shift_xglm_small` file.

Generated embeddings, numerical outputs, logs, and plots are not stored in this
Git repository. The current lab output directory is:

`/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/AuriStream-alpha/AI_OUTPUT/naturalstories_reproduction/xglm_small_3shift/`

## Result

The full reproduction completed successfully on 2026-08-24. It compared all
25 XGLM-small levels and all nine held-out stories, for 225 released-versus-
reproduced values.

- Maximum absolute difference: `5.01541915e-07`
- Mean absolute difference: `1.53137231e-07`
- Correlation across released and reproduced values: `0.9999999999987167`
- Prespecified numerical tolerance: `1e-05` (passed)
- Released and reproduced peak: layer 15
- Reproduced peak mean held-out-story Pearson r: `0.4421498787`

The Hugging Face model was `facebook/xglm-564M`, resolved from `main` to commit
`f3059f01b98ccc877c673149e0178c0e957660f9`. The released comparison file had
SHA-256 `a24b88c1e8a694c28ee8b85974230054e1725912e3c94f185645181ab9ab379b`.

The extraction smoke, remaining-story extraction, regression smoke, and full
fit were Slurm jobs `41589774`, `41590039`, `41590565`, and `41591017`,
respectively. The complete generated report is `REPRODUCTION_REPORT.md` in the
external output directory above; detailed comparison values are in
`results/reference_comparison.csv` there.

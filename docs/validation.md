# Validation contract

A release is reproducible only when all of the following checks pass:

1. Zero acceleration recovers the vacuum IMRPhenomD waveform.
2. FSD, TDS, and SPA+PN produce finite outputs on compatible frequency grids.
3. The IMRPhenomD vacuum waveform agrees with the PyCBC/LALSuite reference.
4. The importable-library ASTs of the documented IMRPhenomD modules match
   their pre-translation baselines; the hash report is stored in
   `docs/phenomd_semantic_audit.json`.
5. All six numerical tables and rendered PNG files are present and finite.
6. Fig. 5 is non-spinning and reproduces its locked numerical baseline.
7. The Fisher calculation uses third-order FSD, the ET PSD, SNR 1000, a fixed
   2:1 mass ratio, and non-spinning components.
8. The 4096 Hz/2048 Hz TDS phase comparison passes over the frequency bands
   displayed in Fig. 1: 5–400 Hz for the lower-mass systems and 5–140 Hz for
   the higher-mass systems.
9. The walkthrough notebook executes from top to bottom in a clean kernel.
10. Quick and full clean-environment reproductions pass, and the release
    manifest exactly matches the exported files.

The latest full-run measurements and check results are stored in
`reproduction_record.json` and `docs/validation_status.json`.

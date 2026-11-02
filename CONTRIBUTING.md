# Contributing to omnia-sdk

Thanks for helping. The most useful contributions right now:

1. **Benchmark results from your hardware.** Run `omnia benchmark formats <your files>` and open a
   [benchmark issue](https://github.com/mishel-0/omnia-sdk/issues/new?template=benchmark.yml) with the JSON.
   Results where `.omnia` is slower are just as welcome.
2. **New format readers** (Ventana BIF benchmarks, Zeiss CZI, OME-TIFF, NRRD, MHA...).
3. **Bug reports** with the file format, the command you ran and `omnia doctor` output.

## Set up

```bash
git clone https://github.com/mishel-0/omnia-sdk.git && cd omnia-sdk
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,train,benchmark,dev]"
pytest
```

## Adding a format

- Write the reader in `omnia_sdk/converters.py` and register its extensions in `detect_format`.
- Add a round-trip test in `tests/test_convert.py` that proves pixels come back unchanged. Use a tiny
  synthetic fixture; never commit patient data.
- Strip patient identifiers from metadata (see the DICOM allow-list).
- Run `omnia benchmark formats` on a public sample and paste the numbers in the PR.

## Pull requests

Keep them small and focused, describe what you measured, and make sure `pytest` passes.
If you change the README charts, regenerate them with `python assets/make_readme_art.py` and `python assets/make_brand.py`.

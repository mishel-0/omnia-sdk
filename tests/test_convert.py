"""Format converters: what goes in must come out, and no patient data in between."""
import numpy as np
import pytest

from omnia_sdk import OmniaContainer, convert
from omnia_sdk.converters import UnsupportedFormat, detect_format

nib = pytest.importorskip("nibabel")
pydicom = pytest.importorskip("pydicom")
tifffile = pytest.importorskip("tifffile")


def _volume(shape=(24, 20, 7), seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 2000, shape, dtype=np.int16)


def test_nifti_roundtrip_is_bit_exact(tmp_path):
    vol = _volume()
    img = nib.Nifti1Image(vol, np.diag([0.7, 0.7, 2.5, 1.0]))
    src = tmp_path / "scan.nii.gz"
    nib.save(img, str(src))
    assert detect_format(src) == "nifti"

    stats = convert(src, tmp_path / "scan.omnia")
    assert stats["format"] == "nifti" and stats["tiles"] == 7
    with OmniaContainer(tmp_path / "scan.omnia") as c:
        assert c.dtype == np.int16
        for i in range(7):
            np.testing.assert_array_equal(c.get_slice(i), vol[:, :, i])
        v = c._meta["volume"]
        assert v["spacing"] == pytest.approx([0.7, 0.7, 2.5])


def _write_dicom(path, arr, z, instance, name="DOE^JANE"):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = name
    ds.PatientID = "SECRET-123"
    ds.PatientBirthDate = "19700101"
    ds.StudyDate = "20240101"
    ds.Modality = "CT"
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0, 0, z]
    ds.PixelSpacing = [0.8, 0.8]
    ds.SliceThickness = 2.0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_dicom_series_is_sorted_exact_and_anonymous(tmp_path):
    vol = _volume((16, 12, 5), seed=9)
    d = tmp_path / "ct"
    d.mkdir()
    order = [3, 0, 4, 1, 2]  # written out of order on purpose
    for k, z in enumerate(order):
        _write_dicom(d / f"{k}.dcm", vol[:, :, z], z=float(z) * 2.0, instance=k)
    assert detect_format(d) == "dicom-series"

    convert(d, tmp_path / "ct.omnia")
    with OmniaContainer(tmp_path / "ct.omnia") as c:
        for z in range(5):
            np.testing.assert_array_equal(c.get_slice(z), vol[:, :, z])
        v = c._meta["volume"]
        assert v["Modality"] == "CT" and v["RescaleIntercept"] == -1024.0
    raw = (tmp_path / "ct.omnia").read_bytes()
    import json, zstandard
    # decode metadata block and make sure nothing identifying is in it
    meta_size = int.from_bytes(raw[8:16], "little")
    meta_text = zstandard.ZstdDecompressor().decompress(raw[16:16 + meta_size]).decode()
    for secret in ("DOE", "SECRET-123", "19700101", "20240101"):
        assert secret not in meta_text
    json.loads(meta_text)


def test_tiled_tiff_slide_roundtrip(tmp_path):
    openslide = pytest.importorskip("openslide")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    src = tmp_path / "slide.tif"
    tifffile.imwrite(str(src), img, tile=(256, 256), photometric="rgb", compression=None)
    assert detect_format(src) == "whole-slide"
    stats = convert(src, tmp_path / "slide.omnia", min_level=0)
    assert stats["tiles"] == 4
    with OmniaContainer(tmp_path / "slide.omnia") as c:
        np.testing.assert_array_equal(c.get_slice(0), img[:256, :256])


def test_unsupported_input_is_a_clear_error(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(UnsupportedFormat):
        detect_format(f)

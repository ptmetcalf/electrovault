from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from photo_brain.index import ExifDataRow, PhotoFileRow, init_db, session_factory
from photo_brain.ingest import ingest_directory, read_exif, scan_photos


def _make_image(path: Path, with_exif: bool = False) -> None:
    img = Image.new("RGB", (10, 10), color="red")
    if with_exif:
        exif = Image.Exif()
        exif[36867] = "2021:01:02 03:04:05"
        exif[306] = "2021:01:02 03:04:05"  # fallback DateTime
        exif[271] = "TestMake"
        exif[272] = "TestModel"
        exif[305] = "UnitTestSoftware"
        exif[274] = 1  # Orientation
        exif[33434] = (1, 60)  # ExposureTime 1/60s
        exif[33437] = (4, 1)  # FNumber f/4
        exif[34855] = 200  # ISO
        exif[37386] = (35, 1)  # FocalLength 35mm
        exif[42036] = "TestLens"
        img.save(path, exif=exif)
    else:
        img.save(path)


def test_scan_photos_reads_files(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.png"
    _make_image(first)
    _make_image(second)
    (tmp_path / "ignore.txt").write_text("skip me")

    photos = scan_photos(tmp_path)
    ids = {p.id for p in photos}
    assert len(photos) == 2
    assert all(len(i) == 64 for i in ids)


def test_read_exif_parses_datetime(tmp_path: Path) -> None:
    file_path = tmp_path / "with_exif.jpg"
    _make_image(file_path, with_exif=True)

    exif = read_exif(file_path)
    assert exif.datetime_original is not None
    assert exif.datetime_original.year == 2021
    assert exif.camera_make == "TestMake"
    assert exif.camera_model == "TestModel"
    assert exif.software == "UnitTestSoftware"
    assert exif.orientation == 1
    assert exif.iso == 200
    assert exif.focal_length == 35.0
    assert exif.f_number == 4.0
    assert exif.exposure_time == 1 / 60
    assert exif.lens_model == "testlens"


def test_ingest_directory_upserts(tmp_path: Path) -> None:
    photo_path = tmp_path / "photo.jpg"
    _make_image(photo_path, with_exif=True)
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)

    with SessionLocal() as session:
        ingest_directory(tmp_path, session)
        stored = session.scalars(select(PhotoFileRow)).all()
        assert len(stored) == 1
        exif_row = session.scalar(select(ExifDataRow))
        assert exif_row is not None
        assert exif_row.datetime_original.year == 2021

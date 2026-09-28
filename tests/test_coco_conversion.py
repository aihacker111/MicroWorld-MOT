import json
from pathlib import Path

from PIL import Image

from tools.convert_tracking_to_coco import convert_split, main


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_mot_sequence(root: Path, name: str, rows: list[str]) -> None:
    sequence = root / "train" / name
    _write_image(sequence / "img1" / "000001.jpg")
    (sequence / "gt").mkdir(parents=True)
    (sequence / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        "frameRate=25\n"
        "seqLength=1\n"
        "imWidth=100\n"
        "imHeight=80\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )


def test_mot17_deduplicates_detector_copies_and_keeps_tracking_fields(tmp_path: Path) -> None:
    rows = [
        "1,7,-5,10,20,30,1,1,0.75",
        "1,8,10,10,20,30,1,3,1.0",
        "1,9,10,10,20,30,0,1,1.0",
    ]
    _write_mot_sequence(tmp_path, "MOT17-02-DPM", rows)
    _write_mot_sequence(tmp_path, "MOT17-02-FRCNN", rows)
    _write_mot_sequence(tmp_path, "MOT17-02-SDP", rows)

    coco = convert_split("mot17", tmp_path, "train", category_mode="person")

    assert [video["name"] for video in coco["videos"]] == ["MOT17-02"]
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    annotation = coco["annotations"][0]
    assert annotation["track_id"] == 7
    assert annotation["visibility"] == 0.75
    assert annotation["bbox"] == [0.0, 10.0, 15.0, 30.0]


def test_visdrone_person_mode_filters_ignored_and_vehicle_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "VisDrone2019-MOT-train"
    _write_image(dataset_root / "sequences" / "uav0001" / "0000001.jpg")
    annotations = dataset_root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "uav0001.txt").write_text(
        "1,11,10,10,20,20,1,1,0,2\n"
        "1,12,20,20,20,20,1,2,1,1\n"
        "1,13,20,20,20,20,1,4,0,0\n"
        "1,14,20,20,20,20,0,1,0,0\n"
        "1,15,20,20,20,20,1,11,0,0\n",
        encoding="utf-8",
    )

    coco = convert_split("visdrone", tmp_path, "train", category_mode="person")

    assert len(coco["videos"]) == 1
    assert [item["track_id"] for item in coco["annotations"]] == [11, 12]
    assert [item["category_id"] for item in coco["annotations"]] == [1, 1]
    assert coco["annotations"][0]["occlusion"] == 2
    assert coco["annotations"][0]["visibility"] == 0.0


def test_cli_writes_compact_json(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "DanceTrack"
    _write_mot_sequence(dataset_root, "dancetrack0001", ["1,3,1,2,10,12,1,1,1"])
    output = tmp_path / "coco"
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_tracking_to_coco.py",
            "--dataset",
            "dancetrack",
            "--root",
            str(dataset_root),
            "--output",
            str(output),
            "--splits",
            "train",
        ],
    )

    main()

    converted = json.loads((output / "dancetrack_train.json").read_text(encoding="utf-8"))
    assert converted["images"][0]["frame_id"] == 1
    assert converted["annotations"][0]["track_id"] == 3

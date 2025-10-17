import csv
from pathlib import Path

class CSVLogger:
    def __init__(self,
                 base_dir: str,
                 date: str,
                 fieldnames: dict):
        """
        base_dir: 로그를 저장할 최상위 디렉토리
        date: 하위 폴더 이름
        fieldnames: {
            'log': ["timestep", "vehicle id", ...],
            'plog': ["timestep", "person id", ...],
            ...
        }
        """
        self.handles = {}
        save_path = Path(base_dir) / date
        save_path.mkdir(parents=True, exist_ok=True)

        for name, header in fieldnames.items():
            f = open(save_path / f"{name}.csv", "a", newline="", encoding="utf-8")
            writer = csv.writer(f)

            # Write header when the file is empty
            if f.tell() == 0:
                writer.writerow(header)
            self.handles[name] = (f, writer)

    def log(self, name: str, row: list):
        f, writer = self.handles[name]
        writer.writerow(row)
        f.flush()

    def close(self):
        for f, _ in self.handles.values():
            f.close()
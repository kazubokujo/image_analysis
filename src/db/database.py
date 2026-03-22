import sqlite3
import pickle
import os
from typing import List, Tuple

from config import DB_PATH


class ImageDatabase:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            feature BLOB,
            label TEXT,
            face_id INTEGER,
            face_feature BLOB,
            organized INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_path ON images(path)
        """)

        self.conn.commit()

    # ----------------------------
    # INSERT
    # ----------------------------
    def insert_image(self, path: str, feature):
        blob = pickle.dumps(feature)

        self.conn.execute("""
        INSERT OR IGNORE INTO images (path, feature)
        VALUES (?, ?)
        """, (path, blob))

        self.conn.commit()

    # ----------------------------
    # GET
    # ----------------------------
    def get_all_features(self) -> Tuple[List, List]:
        cursor = self.conn.execute("""
        SELECT feature, label FROM images
        WHERE feature IS NOT NULL AND label IS NOT NULL
        """)

        features = []
        labels = []

        for f_blob, label in cursor.fetchall():
            features.append(pickle.loads(f_blob))
            labels.append(label)

        return features, labels

    def get_unlabeled(self):
        return self.conn.execute("""
        SELECT path, feature FROM images
        WHERE label IS NULL
        """).fetchall()

    def exists(self, path: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM images WHERE path=?",
            (path,)
        )
        return cur.fetchone() is not None

    # ----------------------------
    # UPDATE
    # ----------------------------
    def update_label(self, path: str, label: str):
        self.conn.execute("""
        UPDATE images SET label=?
        WHERE path=?
        """, (label, path))

        self.conn.commit()

    def update_face_id(self, path: str, face_id: int):
        self.conn.execute("""
        UPDATE images SET face_id=?
        WHERE path=?
        """, (face_id, path))

        self.conn.commit()

    # ----------------------------
    # DEBUG / 管理
    # ----------------------------
    def count(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM images")
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()
        
        
    def get_unorganized(self):
        return self.conn.execute("""
        SELECT path, label FROM images
        WHERE label IS NOT NULL AND organized=0
        """).fetchall()


    def mark_organized(self, path):
        self.conn.execute("""
        UPDATE images SET organized=1 WHERE path=?
        """, (path,))
        self.conn.commit()
        
    
    def get_labeled_features(self):
        cursor = self.conn.execute("""
        SELECT feature, label FROM images
        WHERE label IS NOT NULL
        """)

        import pickle

        features = []
        labels = []

        for f_blob, label in cursor.fetchall():
            features.append(pickle.loads(f_blob))
            labels.append(label)

        return features, labels
    
    
    def get_faces(self):
        cursor = self.conn.execute("""
        SELECT face_id, face_feature FROM images
        WHERE face_feature IS NOT NULL
        """)

        import pickle

        data = []
        for face_id, blob in cursor.fetchall():
            data.append((face_id, pickle.loads(blob)))

        return data


    def update_face(self, path, face_id, face_feature):
        import pickle

        blob = pickle.dumps(face_feature)

        self.conn.execute("""
        UPDATE images SET face_id=?, face_feature=?
        WHERE path=?
        """, (face_id, blob, path))

        self.conn.commit()
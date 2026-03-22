import os
import time
from tqdm import tqdm

# from face.face import extract_face_encoding, face_distance
from classify.classifier import Classifier
from config import INPUT_DIR
from db.database import ImageDatabase
from feature.extractor import FeatureExtractor
from organizer.organizer import organize_file


def get_all_files():
    files = []
    for root, _, filenames in os.walk(INPUT_DIR):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                files.append(os.path.join(root, f))
    return files


def main():
    print("=== Image AI System Start ===")

    db = ImageDatabase()
    extractor = FeatureExtractor()

    files = get_all_files()

    print(f"Found {len(files)} files")


    for i, path in enumerate(tqdm(files)):
        if i % 500 == 0:
            print(f"[EXTRACT] {i} / {len(files)}")

        if db.exists(path):
            continue

        feature = extractor.extract(path)

        if feature is not None:
            db.insert_image(path, feature)

        if i % 100 == 0:
            db.conn.commit()


    print("=== Classification Start ===")

    classifier = Classifier(extractor.model, extractor.processor, db)

    unlabeled = db.get_unlabeled()

    print(f"Unlabeled: {len(unlabeled)}")

    from config import BATCH_SIZE
    import pickle

    for i in range(0, len(unlabeled), BATCH_SIZE):
        batch = unlabeled[i:i+BATCH_SIZE]

        print(f"[CLASSIFY] {i} / {len(unlabeled)}")

        for path, feature_blob in batch:
            feature = pickle.loads(feature_blob)
            feature = feature.flatten()

            label = classifier.classify(path, feature)
            db.update_label(path, label)

        db.conn.commit()

    print("=== Classification Done ===")
    
    
    print("=== Organizing Start ===")

    cursor = db.conn.execute("""
    SELECT path, label, face_id FROM images WHERE organized=0
    """)

    for path, label, face_id in cursor.fetchall():
        organize_file(path, label, face_id)
        db.mark_organized(path)

        
    print("=== Organizing Done ===")       
    
    
    
    

    
    
    import pickle
    from person.cluster import cluster_people

    print("=== Person Clustering Start ===")

    cursor = db.conn.execute("""
    SELECT path, feature FROM images WHERE label LIKE '%person%'
    """)

    rows = cursor.fetchall()

    paths = []
    features = []

    for path, blob in rows:
        f = pickle.loads(blob)

        f = f.flatten()

        if f.shape[0] < 100:
            continue

        paths.append(path)
        features.append(f)

    cluster_ids = []

    if len(features) > 0:
        cluster_ids = cluster_people(features)

        for path, cid in zip(paths, cluster_ids):
            db.conn.execute("""
            UPDATE images SET face_id=? WHERE path=?
            """, (cid, path))

        db.conn.commit()

    print("=== Person Clustering Done ===")
    
    
 
    
    
    

    

    # print("=== Face Processing Start ===")

    # faces_db = db.get_faces()
    # next_face_id = len(faces_db) + 1

    # for path, _ in unlabeled:
    #     face = extract_face_encoding(path)

    #     if face is None:
    #         continue

    #     assigned = None

    #     for fid, db_face in faces_db:
    #         if face_distance(face, db_face) < 0.5:
    #             assigned = fid
    #             break

    #     if assigned is None:
    #         assigned = next_face_id
    #         next_face_id += 1

    #     db.update_face(path, assigned, face)

    # print("=== Face Processing Done ===")    


    print("=== Done ===")


if __name__ == "__main__":
    main()
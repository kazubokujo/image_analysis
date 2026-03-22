from db.database import ImageDatabase

db = ImageDatabase()

print("登録数:", db.count())